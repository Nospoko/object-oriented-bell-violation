import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from physics.particles import OneBitEntangledPhoton, LocalDeterministicPhoton


def main():
    st.write("# Local Photon Model")
    df = run_local_experiment()
    df["angle_bin"] = df.angle_diff.round(2)
    # TODO: Write an experiment results wrapper class & nice chart
    angle_agreement = df.groupby("angle_bin").agreement.mean().reset_index()

    fig = draw_polarization_agreement_chart(agreement_df=angle_agreement)
    st.pyplot(fig)

    st.write("# 1-bit superluminal communication")
    df = run_entangled_experiment()
    df["angle_bin"] = df.angle_diff.round(2)
    # TODO: Write an experiment results wrapper class & nice chart
    angle_agreement = df.groupby("angle_bin").agreement.mean().reset_index()

    fig = draw_polarization_agreement_chart(agreement_df=angle_agreement)
    st.pyplot(fig)


def draw_polarization_agreement_chart(agreement_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=[9, 4])
    ax.plot(
        agreement_df.angle_bin,
        agreement_df.agreement,
        "--.",
        ms=7,
        label="photon model",
        color="indigo",
    )

    ax.plot(
        agreement_df.angle_bin,
        np.cos(agreement_df.angle_bin) ** 2,
        label="empirical reality",
        color="crimson",
    )
    x = agreement_df.angle_bin.values
    y_down = np.zeros_like(x)
    y_up = np.ones_like(x)
    y_diagonal = np.linspace(1, 0, len(x))
    ax.fill_between(x, y_down, y_diagonal, color="lime", alpha=0.2)
    ax.text(0.25, 0.3, "Bell local", fontsize=16, rotation=-15)

    ax.fill_between(x, y_diagonal, y_up, color="teal", alpha=0.2)
    ax.text(1.0, 0.6, "Bell non-local", fontsize=16, rotation=-15)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, x.max())

    ax.legend()

    return fig


def run_entangled_experiment() -> pd.DataFrame:
    results = []
    for it in range(500_000):
        reference_angle = np.random.random() * np.pi
        a_photon = OneBitEntangledPhoton(reference_angle)
        b_photon = OneBitEntangledPhoton(reference_angle)

        a_photon.entangle(b_photon)
        b_photon.entangle(a_photon)

        # We are interested in the angle difference between detectors
        # and we only want to measure within range of [0 - pi/2] ...
        a_detector_angle = 0
        # ... so we only allow the movement of the second detector within that range
        b_detector_angle = np.random.random() * np.pi / 2

        a_polarization_status = a_photon.measure_polarization(a_detector_angle)
        b_polarization_status = b_photon.measure_polarization(b_detector_angle)
        result = {
            "a_outcome": a_polarization_status.value,
            "b_outcome": b_polarization_status.value,
            "a_detector": a_detector_angle,
            "b_detector": b_detector_angle,
        }
        results.append(result)

    df = pd.DataFrame(results)
    df["agreement"] = df.a_outcome == df.b_outcome
    df["angle_diff"] = df.b_detector - df.a_detector

    return df


def run_local_experiment() -> pd.DataFrame:
    results = []
    for it in range(500_000):
        # In the experimental setup we have photons with same polarization
        polarization_angle = np.random.random() * np.pi
        a_photon = LocalDeterministicPhoton(polarization_angle)
        b_photon = LocalDeterministicPhoton(polarization_angle)

        # We are interested in the angle difference between detectors
        # and we only want to measure within range of [0 - pi/2] ...
        a_detector_angle = 0
        # ... so we only allow the movement of the second detector within that range
        b_detector_angle = np.random.random() * np.pi / 2

        # Without entanglement order of measurement doesn't matter
        # (does it matter with entangled particles? :thinking:)
        a_polarization_status = a_photon.measure_polarization(a_detector_angle)
        b_polarization_status = b_photon.measure_polarization(b_detector_angle)
        result = {
            "a_outcome": a_polarization_status.value,
            "b_outcome": b_polarization_status.value,
            "a_detector": a_detector_angle,
            "b_detector": b_detector_angle,
        }
        results.append(result)

    df = pd.DataFrame(results)
    df["agreement"] = df.a_outcome == df.b_outcome
    df["angle_diff"] = df.b_detector - df.a_detector

    return df


if __name__ == "__main__":
    main()
