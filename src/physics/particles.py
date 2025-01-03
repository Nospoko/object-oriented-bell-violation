from enum import Enum
from abc import ABC, abstractmethod

import numpy as np


class PolarizationMeasurementOutcome(Enum):
    PASSED = True
    ABSORBED = False


class PolarizedParticleBase(ABC):
    @abstractmethod
    def measure_polarization(self, detector_angle_rad: float) -> PolarizationMeasurementOutcome:
        pass


class EntangledPolarizedParticleBase(PolarizedParticleBase):
    @abstractmethod
    def superluminal_communication(self):
        pass


class LocalDeterministicPhoton(PolarizedParticleBase):
    def __init__(self, polarization_angle: float):
        self.polarization_angle = polarization_angle

    def measure_polarization(self, detector_angle_rad: float) -> PolarizationMeasurementOutcome:
        angle_difference = self.polarization_angle - detector_angle_rad
        # This means the difference is within [-pi/4, pi/4]
        if np.cos(angle_difference) ** 2 > 0.5:
            return PolarizationMeasurementOutcome.PASSED
        else:
            # And this means it's more than pi/4 and less than 3pi/4
            return PolarizationMeasurementOutcome.ABSORBED


class OneBitEntangledPhoton(EntangledPolarizedParticleBase):
    def __init__(
        self,
        reference_angle_rad: float,
    ):
        self.decided = False
        self.use_strategy_b = False

        # Entangled photons do not have polarization decided when created
        # but they share a reference frame
        self.reference_angle_rad = reference_angle_rad

    def strategy_a(self, detector_angle_rad: float) -> PolarizationMeasurementOutcome:
        angle_difference = self.reference_angle_rad - detector_angle_rad

        # This means the difference is within [-pi/4, pi/4]
        if np.cos(angle_difference) ** 2 > 0.5:
            return PolarizationMeasurementOutcome.PASSED
        else:
            # And this means it's more than pi/4 and less than 3pi/4
            return PolarizationMeasurementOutcome.ABSORBED

    def strategy_b(self, detector_angle_rad: float) -> PolarizationMeasurementOutcome:
        angle_difference = self.reference_angle_rad - detector_angle_rad

        # This means the difference is within [-pi/4, pi/4]
        if np.cos(angle_difference - np.pi / 4) ** 2 > 0.5:
            return PolarizationMeasurementOutcome.PASSED
        else:
            # And this means it's more than pi/4 and less than 3pi/4
            return PolarizationMeasurementOutcome.ABSORBED

    def entangle(self, other_photon: "OneBitEntangledPhoton"):
        self.other_photon = other_photon

    def superluminal_communication(self, use_strategy_b: bool):
        self.use_strategy_b = use_strategy_b
        self.decided = True

    def measure_polarization(self, detector_angle_rad: float) -> PolarizationMeasurementOutcome:
        # Quantum magic
        if not self.decided:
            angle_difference = self.reference_angle_rad - detector_angle_rad
            self.use_strategy_b = (angle_difference - np.pi / 8) % (np.pi / 2) < np.pi / 4
            self.other_photon.superluminal_communication(use_strategy_b=self.use_strategy_b)

        if self.use_strategy_b:
            return self.strategy_b(detector_angle_rad)
        else:
            return self.strategy_a(detector_angle_rad)
