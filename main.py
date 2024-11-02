import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from physics.particles import OneBitEntangledPhoton, LocalDeterministicPhoton


def main():
    st.write("# Object Oriented Bell Violation")
    st.write(
        """
    This is a part of the coding challenge for [EPR Labs](https://epr-labs.com) data-science internship for physicists.

    The challenge is to implement a non-local model of a pair of entangled particles that recreates
    the results of Bell-violating EPR experiments through a Monte Carlo simulation.
    """
    )

    st.write(
        """
    Particle class should inherid `PolarizedParticleBase` – for simplicity you can assume
    it's a photon, and we're interested in measuring polarization given a detector at an angle.
    """
    )

    code = """
    class PolarizedParticleBase(ABC):
        @abstractmethod
        def measure_polarization(self, detector_angle_rad: float) -> PolarizationMeasurementOutcome:
            pass
    """
    st.code(code, language="python")

    st.write(
        """
    The Monte Carlo simulation should perform at least 10k measurements of an entangled
    pair with random detector angles. We're interested in tracking the information about
    detector angles and measurement outputs. Here's an overview of what the simulation loop can look like:
    """
    )

    code = """
    results = []
    for it in range(100_000):
        a_photon = YourPhoton(...)
        b_photon = YourPhoton(...)

        # Your entanglement mechanism
        a_photon.entangle(b_photon)

        # We are interested only in the angle difference in range [0, pi/2]
        a_detector_angle = 0
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
    """
    st.code(code, language="python")

    st.write(
        """
    Below you can see results for two different simulations.
    One for a local hidden variables model, which fails to violate Bell inequalities,
    and one for a model with 1-bit non-local communication, that violates Bell inequalities, but
    fails to reproduce empirical results.
    """
    )

    st.write("## Local Photon Model")
    st.write("What Einstein hoped for 🥲")
    df = run_local_experiment()
    df["angle_bin"] = df.angle_diff.round(2)
    # TODO: Write an experiment results wrapper class & nice chart
    angle_agreement = df.groupby("angle_bin").agreement.mean().reset_index()

    fig = draw_polarization_agreement_chart(agreement_df=angle_agreement)
    st.pyplot(fig)

    st.write("## 1-bit superluminal communication")
    st.write(
        "This simulation recreates a model proposed by T.Maudlin in 1992 [[1](https://www.jstor.org/stable/192771)]."
    )
    df = run_entangled_experiment()
    df["angle_bin"] = df.angle_diff.round(2)
    # TODO: Write an experiment results wrapper class & nice chart
    angle_agreement = df.groupby("angle_bin").agreement.mean().reset_index()

    fig = draw_polarization_agreement_chart(agreement_df=angle_agreement)
    st.pyplot(fig)

    show_refs()


def show_refs():
    title = "The Communication Cost of Simulating Bell Correlations"
    link = "https://arxiv.org/abs/quant-ph/0304076"
    st.write(f"[{title}]({link})")


def draw_polarization_agreement_chart(agreement_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=[9, 4])
    x = agreement_df.angle_bin.values * 180 / np.pi
    ax.plot(
        x,
        agreement_df.agreement,
        ".",
        ms=7,
        label="this simulation",
        color="indigo",
    )

    ax.plot(
        x,
        np.cos(agreement_df.angle_bin) ** 2,
        label="EPR experiments",
        color="teal",
    )
    ax.set_xlabel("Detectors angle difference [degrees]", fontsize=14)
    ax.set_ylabel(
        "Proportion of agreement in\nspace time separated\npolarization angle measurements",
        fontsize=14,
    )

    ax.set_ylim(0, 1)
    ax.set_xlim(0, x.max())

    ax.legend()

    y_down = np.zeros_like(x)
    # y_up = np.ones_like(x)
    y_diagonal = np.linspace(1, 0, len(x))
    ax.fill_between(x, y_down, y_diagonal, color="gray", alpha=0.2)
    ax.text(10.3, 0.3, "Possible with\nBell-local theory", fontsize=20, rotation=0)

    # ax.fill_between(x, y_diagonal, y_up, color="gray", alpha=0.4)
    ax.text(54.3, 0.50, "Not possible with\nBell-local theory", fontsize=20, rotation=0)

    return fig


def run_entangled_experiment(n_steps: int = 100_000) -> pd.DataFrame:
    results = []
    for it in range(n_steps):
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


def run_local_experiment(n_steps: int = 100_000) -> pd.DataFrame:
    results = []
    for it in range(n_steps):
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
