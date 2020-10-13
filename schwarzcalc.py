import numpy as np

# make universal constants

c = 2.99e8 * 1e3
amu2eV = 931.49432e6 / c**2
kT = 23.5e-3
pi = np.pi

class RFCarpet:
    def __init__(self, K0, Q, mBa, pp0, Vrf, Ep):
        self.k0 = K0  # mm^2/V/s
        self.q = Q  # e
        self.mba = mBa  # amu
        self.pp0 = pp0  # unitless
        self.vrf = Vrf  # V (amplitude)
        self.ep = Ep  # V/mm
    
        self.ohm = 13.56e6  # Hz
        self.t0t = 1.0  # unitless
        self.gap = 0.08  # mm
        self.pitch = 0.16  # mm
        self.gamma = self.gap/self.pitch  # unitless
        
        self.mi = self.mba * amu2eV  # eV*s^2/mm^s
        self.k = self.k0 / self.pp0 / self.t0t  # mm^2/V/s
        self.damp = self.q / self.mi / self.k  # Hz

    def recalc_vars(self):
        self.gamma = self.gap/self.pitch  # unitless
        
        self.mi = self.mba * amu2eV  # eV*s^2/mm^s
        self.k = self.k0 / self.pp0 / self.t0t  # mm^2/V/s
        self.damp = self.q / self.mi / self.k  # Hz

    def ymin(self, nphase=2):
        a = nphase/2 * self.pitch
        return -a / (2*pi) * np.log(self.ep * a * (self.ohm**2 + self.damp**2) \
                * pi/(8 * np.sin(pi * self.gamma/2)**2) * self.mi/self.q \
                * (self.gamma * a/(2*2*self.vrf))**2)

    def potential_schwarz(self, y, nphase=2):
        a = nphase/2 * self.pitch
        v0 = self.q / (4 * self.mi) / (self.ohm**2 + self.damp**2)
        return self.ep*y + v0 * (8 * (2*self.vrf)/(self.gamma * a * pi))**2 \
                * np.sin(pi * self.gamma/2)**2 * np.exp(-2*pi*y/a)

