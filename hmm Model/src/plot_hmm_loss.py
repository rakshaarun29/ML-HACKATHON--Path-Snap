"""Plot hyperparameter error/loss curves (1 - F1) and Viterbi NLL for HMM_Smoothed."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline_hmm import load_sample, project_to_meters, hmm_map_match, evaluate_route


def eval_params(samples, sigma_z: float, beta: float):
    cnt_f1s, len_f1s = [], []
    for s in samples:
        n, a, t, r = load_sample(s)
        xy_n, xy_t = project_to_meters(n, t[:, :2])
        pred, alens = hmm_map_match(xy_n, a, xy_t, timestamps=t[:, 2], sigma_z=sigma_z, beta=beta)
        m = evaluate_route(pred, r, alens)
        cnt_f1s.append(m["f1_cnt"])
        len_f1s.append(m["f1_len"])
    return (1.0 - float(np.mean(cnt_f1s))) * 100.0, (1.0 - float(np.mean(len_f1s))) * 100.0


def main():
    pkg_root = Path(__file__).resolve().parent.parent
    data_root = pkg_root.parent
    samples = sorted([d for d in (data_root / "Student folder" / "student_test").iterdir() if d.is_dir()])

    sigmas = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 35.0]
    betas = [5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 75.0]

    sig_cnt_loss, sig_len_loss = [], []
    for sz in sigmas:
        cl, ll = eval_params(samples, sigma_z=sz, beta=25.0)
        sig_cnt_loss.append(cl)
        sig_len_loss.append(ll)

    beta_cnt_loss, beta_len_loss = [], []
    for b in betas:
        cl, ll = eval_params(samples, sigma_z=15.0, beta=b)
        beta_cnt_loss.append(cl)
        beta_len_loss.append(ll)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax1.plot(sigmas, sig_len_loss, "o-", linewidth=2, color="#2ecc71", label="Len-F1 Loss (1 - Len-F1)")
    ax1.plot(sigmas, sig_cnt_loss, "s--", linewidth=2, color="#3498db", label="Cnt-F1 Loss (1 - Cnt-F1)")
    ax1.axvline(15.0, color="gray", linestyle=":", label="Selected sigma_z = 15m")
    ax1.set_title("HMM Loss Curve vs. Emission Noise (sigma_z)\n(fixed beta = 25m)", fontweight="bold")
    ax1.set_xlabel("sigma_z (meters)", fontweight="bold")
    ax1.set_ylabel("Error / Loss (%)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(betas, beta_len_loss, "o-", linewidth=2, color="#2ecc71", label="Len-F1 Loss (1 - Len-F1)")
    ax2.plot(betas, beta_cnt_loss, "s--", linewidth=2, color="#3498db", label="Cnt-F1 Loss (1 - Cnt-F1)")
    ax2.axvline(25.0, color="gray", linestyle=":", label="Selected beta = 25m")
    ax2.set_title("HMM Loss Curve vs. Transition Scale (beta)\n(fixed sigma_z = 15m)", fontweight="bold")
    ax2.set_xlabel("beta (meters)", fontweight="bold")
    ax2.set_ylabel("Error / Loss (%)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    out_path = pkg_root / "report" / "hmm_loss_curve.png"
    plt.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
