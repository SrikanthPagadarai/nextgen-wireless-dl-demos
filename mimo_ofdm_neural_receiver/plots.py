import os
import numpy as np
import matplotlib.pyplot as plt

# Constants
INFILE = os.path.join("results", "all_baseline_results_cdlC.npz")

# Load Data
data = np.load(INFILE, allow_pickle=True)
ebno_db = data["ebno_db"]
directions = data["directions"]
perfect_csi = data["perfect_csi"]
ber = data["ber"]
bler = data["bler"]
cdl_model = str(data["cdl_model"])

os.makedirs("results", exist_ok=True)

loss = np.load("loss_history.npy")
plt.plot(loss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

## baseline plots
OUTFILE_UPLINK = os.path.join("results", "uplink_baseline_ber_cdlC.png")
OUTFILE_DOWNLINK = os.path.join("results", "downlink_baseline_ber_cdlC.png")
for direction, outfile in [("uplink", OUTFILE_UPLINK), ("downlink", OUTFILE_DOWNLINK)]:
    plt.figure()
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BER")
    plt.grid(which="both")
    plt.ylim([1e-4, 1.1])
    if direction == "uplink":
        plt.title(f"4x8 MIMO - {direction.capitalize()} - CDL-{cdl_model}")
    elif direction == "downlink":
        plt.title(f"8x4 MIMO - {direction.capitalize()} - CDL-{cdl_model}")

    for i in range(len(directions)):
        if directions[i] == direction:
            label = f"{'perfect CSI' if perfect_csi[i] else 'imperfect CSI'}"
            plt.semilogy(ebno_db, ber[i], label=label, marker='o', linestyle='-')

    plt.legend()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {direction} plot to {outfile}")

OUTFILE_UPLINK = os.path.join("results", "uplink_baseline_bler_cdlC.png")
OUTFILE_DOWNLINK = os.path.join("results", "downlink_baseline_bler_cdlC.png")
for direction, outfile in [("uplink", OUTFILE_UPLINK), ("downlink", OUTFILE_DOWNLINK)]:
    plt.figure()
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("BLER")
    plt.grid(which="both")
    plt.ylim([1e-3, 1.1])
    if direction == "uplink":
        plt.title(f"4x8 MIMO - {direction.capitalize()} - CDL-{cdl_model}")
    elif direction == "downlink":
        plt.title(f"8x4 MIMO - {direction.capitalize()} - CDL-{cdl_model}")

    for i in range(len(directions)):
        if directions[i] == direction:
            label = f"{'perfect CSI' if perfect_csi[i] else 'imperfect CSI'}"
            plt.semilogy(ebno_db, bler[i], label=label, marker='o', linestyle='-')

    plt.legend()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {direction} plot to {outfile}")
