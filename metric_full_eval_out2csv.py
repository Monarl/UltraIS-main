import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.metrics import calculate_niqe, calculate_psnr, calculate_ssim
from tqdm import tqdm

try:
    import lpips
except ImportError:
    lpips = None


def _image_id(p: Path) -> str:
    stem = p.stem
    return stem.split("-")[0]


def _find_files(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _to_lpips_tensor(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    diff = pred.astype(np.float32) - gt.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))


def _to_scalar(x) -> float:
    arr = np.asarray(x)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM/LPIPS/RMSE/NIQE and save CSV.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory of model output images.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory of GT HR images.")
    parser.add_argument("--csv_path", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--crop_border", type=int, default=2, help="Crop border for PSNR/SSIM.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for LPIPS.")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS even if package is installed.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    if not pred_dir.exists():
        missing.append(f"pred_dir not found: {pred_dir}")
    if not gt_dir.exists():
        missing.append(f"gt_dir not found: {gt_dir}")
    if missing:
        raise FileNotFoundError("; ".join(missing))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_lpips = (not args.skip_lpips) and (lpips is not None)
    if use_lpips:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    else:
        lpips_model = None
        if lpips is None and not args.skip_lpips:
            print("[WARN] lpips package is not installed; LPIPS will be written as NaN.")
            print("[HINT] Install with: pip install lpips")

    niqe_params = Path("basicsr/metrics/niqe_pris_params.npz")
    use_niqe = niqe_params.exists()
    if not use_niqe:
        print(f"[WARN] NIQE params file not found: {niqe_params}. NIQE will be written as NaN.")

    gt_files = _find_files(gt_dir)
    pred_files = _find_files(pred_dir)

    if len(pred_files) == 0:
        raise FileNotFoundError(
            f"No prediction images found in pred_dir: {pred_dir}. "
            "Run basicsr.test first with val.save_img=true."
        )

    gt_map = {_image_id(p): p for p in gt_files}
    rows = []
    for pred_path in tqdm(pred_files, desc="Evaluating"):
        stem = pred_path.stem
        img_id = _image_id(pred_path)

        gt_path = gt_map.get(img_id)
        if gt_path is None:
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if pred is None or gt is None:
            continue

        if lpips_model is not None:
            lp = lpips_model(_to_lpips_tensor(pred, device), _to_lpips_tensor(gt, device))
            lpips_val = float(lp.detach().cpu().item())
        else:
            lpips_val = float("nan")

        rmse_255 = _rmse(pred, gt)
        niqe_val = _to_scalar(calculate_niqe(pred, crop_border=0, input_order="HWC", convert_to="y")) if use_niqe else float("nan")

        row = {
            "imgname": stem,
            "psnr": float(calculate_psnr(pred, gt, crop_border=args.crop_border, test_y_channel=False)),
            "ssim": float(calculate_ssim(pred, gt, crop_border=args.crop_border, test_y_channel=False)),
            "lpips": lpips_val,
            "rmse": rmse_255,
            "rmse_01": rmse_255 / 255.0,
            "niqe": niqe_val,
        }
        rows.append(row)

    fields = ["imgname", "psnr", "ssim", "lpips", "rmse", "rmse_01", "niqe"]
    avg = None
    if rows:
        avg = {k: float(np.nanmean([r[k] for r in rows])) for k in fields if k != "imgname"}

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if avg is not None:
            avg_row = {"imgname": f"__avg__ (n={len(rows)})"}
            avg_row.update(avg)
            writer.writerow(avg_row)
        writer.writerows(rows)

    if avg is not None:
        print("Averages:")
        for k, v in avg.items():
            print(f"  {k}: {v:.6f}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
