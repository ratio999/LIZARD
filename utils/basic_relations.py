# Copyright (C) 2020  Ssohrab Borhanian
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import astropy as ap
import numpy as np
from lal import GreenwichMeanSiderealTime

#from gwbench.utils import MTsun, time_to_rad_earth
GNewton    = 6.6743e-11
cLight     = 2.99792458e8
Msun       = 1.9884099021470415e+30
Mpc        = 3.085677581491367e+22
REarth     = 6.371e6
AU         = 1.4959787066e11
year       = 3.1536e7
hPlanck    = 6.62607015e-34
TEarthSol  = 86400   # solar day    (24h 60m 60s)
TEarthSid  = 86164.1 # sidereal day (23h 56m  4.0905s, found root 86164.09053826332)
TEarth     = TEarthSid
tc_offset  = 25134.55 # offset to tc: tc_gwbench = tc_bilby + tc_offset

#-----convert mass in solar masses to seconds-----
MTsun             = Msun * GNewton / cLight**3
#-----convert mass/distance in solar mass/Mpc to dimensionless-----
strain_fac        = GNewton / cLight**2 * Msun/Mpc
#-----convert seconds to radians with a periodicty of one day-----
time_to_rad_earth = 2 * np.pi / TEarth
def ChirpTime(fs, Mc):
    Mc *= GNewton/cLight**3 * Msun
    fn = np.asarray([fs]) if np.isscalar(fs) else np.asarray(fs)
    return ((5./256.) / (Mc**(5./3.) * (np.pi * fn)**(8./3.)))
#-----GMST and GPS time conversions-----
def gmst_of_gps_time(gps_time, radians=False):
    '''
    Calculate the Greenwich Mean Sidereal Time (GMST) given the GPS time.
    Wrapper of the `lal.GreenwichMeanSiderealTime` function.

    Parameters
    ----------
    gps_time : float
        The GPS time in seconds.
    radians : bool, optional
        If True, return the GMST in radians, otherwise in seconds. Default is False.

    Returns
    -------
    gmst : float
        The Greenwich Mean Sidereal Time in seconds or radians.
    '''
    if radians: return GreenwichMeanSiderealTime(gps_time)
    else:       return GreenwichMeanSiderealTime(gps_time) / time_to_rad_earth


#-----Keplerian velocity formula-----
def vel_kep(f, M):
    '''
    Calculate the Keplerian velocity for a given frequency and mass.

    Parameters
    ----------
    f : np.ndarray
        The frequency in Hz.
    M : float
        The mass in seconds.

    Returns
    -------
    v : np.ndarray
        The velocity.
    '''
    return (np.pi * M * f)**(1./3)

def vel_kep_Msolar(f, M):
    '''
    Calculate the Keplerian velocity for a given frequency and mass.

    Parameters
    ----------
    f : np.ndarray
        The frequency in Hz.
    M : float
        The mass in solar mass.

    Returns
    -------
    v : np.ndarray
        The velocity.
    '''
    return vel_kep(f, M * MTsun)


#-----tau_spa-----
def tau_spa_PN0(f, Mc):
    '''
    Calculate the frequency-dependent time tau_spa (tf) in the stationary-phase-approximation to account for Earth's rotation.
    see Eq. (7) - (9) in https://arxiv.org/abs/1710.05325v2

    This 0PN correction is taken from the above source, see text after Eq. (9),
    which is taken from Eq. (4.19) from the book "Gravitational Waves: Volume 1: Theory and Experiments" by Michele Maggiore.

    Parameters
    ----------
    f : np.ndarray
        The frequency in Hz.
    Mc : float
        The chirp mass in seconds.

    Returns
    -------
    tf : np.ndarray
        The frequency-dependent time in seconds.
    '''
    return 5. / 256 * (MTsun * Mc)**(-5./3) * (np.pi * f)**(-8./3)

def tau_spa_PN35(f, Mc, eta, log=np.log):
    '''
    Calculate the frequency-dependent time tau_spa (tf) in the stationary-phase-approximation to account for Earth's rotation.
    see Eq. (7) - (9) in https://arxiv.org/abs/1710.05325v2

    This 3.5PN correction is taken from Eq. (3.8b) in https://arxiv.org/abs/0907.0700

    Parameters
    ----------
    f : np.ndarray
        The frequency in Hz.
    Mc : float
        The chirp mass in seconds.

    Returns
    -------
    tf : np.ndarray
        The frequency-dependent time in seconds.
    '''
    eta2 = eta*eta
    pi2  = np.pi*np.pi
    M    = MTsun * M_of_Mc_eta(Mc, eta)
    vel  = vel_kep(f, M)
    vel2 = vel**2
    vel4 = vel2**2
    vel6 = vel2*vel4

    return  5. * M / (256 * eta * vel2*vel6) * \
        (1 +
        (743./252 + 11./3 * eta) * vel2 -
        32 / 5 * np.pi * vel*vel2 +
        (3058673./508032 + 5429./504 * eta + 617./72 * eta2) * vel4 -
        (7729./252 - 13./3 * eta) * np.pi * vel*vel4 +
        (-10052469856691./23471078400 + 128./3 * pi2 + 6848./105 * np.euler_gamma +
        (3147553127./3048192 - 451./12 * pi2) * eta - 15211./1728 * eta2 +
        25565./1296 * eta*eta2 + 3424./105 * log(16. * vel2)) * vel6 +
        (-15419335./127008 - 75703./756 * eta + 14809./378 * eta2) * np.pi * vel*vel6)


#-----f_isco-----
def f_isco(M):
    '''
    Calculate the frequency at the innermost stable circular orbit (ISCO) for a given mass.

    Parameters
    -----------
    M : float
        The mass in seconds.

    Returns
    -------
    f_isco : float
        The frequency at the ISCO.
    '''
    return 1./6.**(3./2.)/np.pi/M

def f_isco_Msolar(M):
    '''
    Calculate the frequency at the innermost stable circular orbit (ISCO) for a given mass.

    Parameters
    -----------
    M : float
        The mass in solar mass.

    Returns
    -------
    f_isco : float
        The frequency at the ISCO.
    '''
    # convert to sec
    return f_isco(M * MTsun)


#-----f_ew (early warning frequency upper cutoff)-----
def f_ew(tau_ew, M, eta):
    '''
    Calculate the early warning frequency upper cutoff for a given mass and symmetric mass ratio.

    Parameters
    -----------
    tau_ew : float
        The early warning time in seconds.
    M : float
        The mass in seconds.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    f_ew : float
        The early warning frequency upper cutoff.
    '''
    return (5 * M / 256 / eta / tau_ew)**(3/8) / np.pi / M

def f_ew_Msolar(tau_ew, M, eta):
    '''
    Calculate the early warning frequency upper cutoff for a given mass and symmetric mass ratio.

    Parameters
    -----------
    tau_ew : float
        The early warning time in seconds.
    M : float
        The mass in solar mass.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    f_ew : float
        The early warning frequency upper cutoff.
    '''
    # convert to sec
    return f_ew(tau_ew, M * MTsun, eta)


#-----mass functions-----
def eta_of_q(q):
    '''
    Calculate the symmetric mass ratio (eta) given the mass ratio (q).

    Parameters
    ----------
    q : float
        The mass ratio.

    Returns
    -------
    eta : float
        The symmetric mass ratio.
    '''
    return q / (1 + q)**2

def delta_of_q(q):
    '''
    Calculate the delta mass difference parameter given the mass ratio (q).

    Parameters
    ----------
    q : float
        The mass ratio.

    Returns
    -------
    delta : float
        The mass difference parameter.
    '''
    return (1 - 4 * eta_of_q(q))**0.5

def delta_of_eta(eta):
    '''
    Calculate the mass difference parameter (delta) given the symmetric mass ratio (eta).

    Parameters
    ----------
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    delta : float
        The mass difference parameter.
    '''
    return (1 - 4 * eta)**0.5

def q_of_eta(eta, q_gt_1=True):
    '''
    Calculate the mass ratio (q) given the symmetric mass ratio (eta).

    Parameters
    ----------
    eta : float
        The symmetric mass ratio.
    q_gt_1 : bool, optional
        The mass ratio q is greater than 1. Default is True.

    Returns
    -------
    q : float
        The mass ratio.
    '''
    if q_gt_1: return (1 + delta_of_eta(eta)) / (1 - delta_of_eta(eta))
    else:      return (1 - delta_of_eta(eta)) / (1 + delta_of_eta(eta))

def q_of_m1_m2(m1, m2, q_gt_1=1):
    '''
    Calculate the mass ratio (q) given the component masses (m1, m2).

    Parameters
    ----------
    m1 : float
        Mass of the first binary component.
    m2 : float
        Mass of the second binary component.
    q_gt_1 : bool, optional
        The mass ratio q is greater than 1. Default is True.

    Returns
    -------
    q : float
        The mass ratio.
    '''
    if q_gt_1: return m1 / m2
    else:      return m2 / m1

def M_of_Mc_eta(Mc, eta):
    '''
    Calculate the total mass (M) given the chirp mass (Mc) and symmetric mass ratio (eta).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    M : float
        The total mass.
    '''
    return Mc / eta**0.6

def M_of_Mc_q(Mc, q):
    '''
    Calculate the total mass (M) given the chirp mass (Mc) and mass ratio (q).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    q : float
        The mass ratio.

    Returns
    -------
    M : float
        The total mass.
    '''
    return Mc / eta_of_q(q)**0.6

def Mc_of_M_eta(M, eta):
    '''
    Calculate the chirp mass (Mc) given the total mass (M) and symmetric mass ratio (eta).

    Parameters
    ----------
    M : float
        The total mass.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    Mc : float
        The chirp mass.
    '''
    return M * eta**0.6

def Mc_of_M_q(M, q):
    '''
    Calculate the chirp mass (Mc) given the total mass (M) and mass ratio (q).

    Parameters
    ----------
    M : float
        The total mass.
    q : float
        The mass ratio.

    Returns
    -------
    Mc : float
        The chirp mass.
    '''
    return M * eta_of_q(q)**0.6

def Mc_of_m1_m2(m1, m2):
    '''
    Calculate the chirp mass (Mc) given the component masses (m1, m2).

    Parameters
    ----------
    m1 : float
        Mass of the first component.
    m2 : float
        Mass of the second component.

    Returns
    -------
    Mc : float
        The chirp mass.
    '''
    return Mc_of_M_eta(m1 + m2, eta_of_q(m1 / m2))

def m1_of_M_eta(M, eta):
    '''
    Calculate the mass of the primary object (m1) in a binary system
    given the total mass (M) and the symmetric mass ratio (eta).

    Parameters
    ----------
    M : float
        Total mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.

    Returns
    -------
    m1 : float
        Mass of the primary object (m1).
    '''
    return 0.5 * M * (1 + delta_of_eta(eta))

def m2_of_M_eta(M, eta):
    '''
    Calculate the mass of the secondary object (m2) in a binary system
    given the total mass (M) and the symmetric mass ratio (eta).

    Parameters
    ----------
    M : float
        Total mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.

    Returns
    -------
    m2 : float
        Mass of the secondary object (m2).
    '''
    return 0.5 * M * (1 - delta_of_eta(eta))

def m1_of_Mc_eta(Mc, eta):
    """
    Calculate the component mass m1 of a binary system given the chirp mass Mc and symmetric mass ratio eta.

    Parameters
    ----------
    Mc : float
        Chirp mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.

    Returns
    -------
    m1 : float
        Mass of the primary object (m1).
    """
    return 0.5 * M_of_Mc_eta(Mc, eta) * (1 + delta_of_eta(eta))

def m2_of_Mc_eta(Mc, eta):
    """
    Calculate the component mass m2 of a binary system given the chirp mass Mc and symmetric mass ratio eta.

    Parameters
    ----------
    Mc : float
        Chirp mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.

    Returns
    -------
    m2 : float
        Mass of the secondary object (m2).
    """
    return 0.5 * M_of_Mc_eta(Mc, eta) * (1 - delta_of_eta(eta))

def m1_of_M_q(M, q):
    '''
    Calculate the mass of the primary object (m1) in a binary system
    given the total mass (M) and the mass ratio (q).

    Parameters
    ----------
    M : float
        Total mass of the binary system.
    q : float
        Mass ratio of the binary system.

    Returns
    -------
    m1 : float
        Mass of the primary object (m1).
    '''
    return 0.5 * M * (1 + delta_of_q(q))

def m2_of_M_q(M, q):
    '''
    Calculate the mass of the secondary object (m2) in a binary system
    given the total mass (M) and the mass ratio (q).

    Parameters
    ----------
    M : float
        Total mass of the binary system.
    q : float
        Mass ratio of the binary system.

    Returns
    -------
    m2 : float
        Mass of the secondary object (m2).
    '''
    return 0.5 * M * (1 - delta_of_q(q))

def m1_of_Mc_q(Mc, q):
    '''
    Calculate the mass of the primary object (m1) in a binary system
    given the chirp mass (Mc) and the mass ratio (q).

    Parameters
    ----------
    Mc : float
        The chirp mass of the binary system.
    q : float
        The mass ratio of the binary system.

    Returns
    -------
    m1 : float
        Mass of the primary object (m1).
    '''
    return 0.5 * M_of_Mc_q(Mc, q) * (1 + delta_of_q(q))

def m2_of_Mc_q(Mc, q):
    '''
    Calculate the mass of the secondary object (m2) in a binary system
    given the chirp mass (Mc) and the mass ratio (q).

    Parameters
    ----------
    Mc : float
        The chirp mass of the binary system.
    q : float
        The mass ratio of the binary system.

    Returns
    -------
    m2 : float
        Mass of the secondary object (m2).
    '''
    return 0.5 * M_of_Mc_q(Mc, q) * (1 - delta_of_q(q))

def m1_m2_of_M_eta(M, eta):
    '''
    Calculate the primary and secondary masses (m1, m2) of a binary system
    given the total mass M and symmetric mass ratio eta.

    Parameters
    ----------
    M : float
        Total mass.
    eta : float
        Symmetric mass ratio.

    Returms
    -------
    m1 : float
        Mass of the primary object.
    m2 : float
        Mass of the secondary object.
    '''
    delta = delta_of_eta(eta)
    return 0.5 * M * (1 + delta), 0.5 * M * (1 - delta)

def m1_m2_of_Mc_eta(Mc, eta):
    '''
    Calculate the primary and secondary masses (m1, m2) of a binary system
    given the chirp mass (Mc) and symmetric mass ratio (eta).

    Parameters
    ----------
    Mc : float
        Chirp mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.

    Returns
    -------
    m1 : float
        Mass of the primary object.
    m2 : float
        Mass of the secondary object.
    '''
    return m1_m2_of_M_eta(M_of_Mc_eta(Mc, eta), eta)

def M_eta_of_m1_m2(m1,m2):
    '''
    Calculate the total mass (M) and symmetric mass ratio (eta) of a binary system
    given the component masses (m1, m2).

    Parameters
    ----------
    m1 : float
        Mass of the primary object.
    m2 : float
        Mass of the secondary object.

    Returns
    -------
    M : float
        Total mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.
    '''
    return m1+m2, eta_of_q(m1/m2)

def Mc_eta_of_M_q(M, q):
    '''
    Calculate the chirp mass (Mc) and symmetric mass ratio (eta) of a binary system
    given the total mass (M) and mass ratio (q).

    Parameters
    ----------
    M : float
        Total mass of the binary system.
    q : float
        Mass ratio of the binary system.

    Returns
    -------
    Mc : float
        Chirp mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.
    '''
    eta = eta_of_q(q)
    return Mc_of_M_eta(M, eta), eta

def Mc_eta_of_m1_m2(m1,m2):
    '''
    Calculate the chirp mass (Mc) and symmetric mass ratio (eta) of a binary system
    given the component masses (m1, m2).

    Parameters
    ----------
    m1 : float
        Mass of the primary object.
    m2 : float
        Mass of the secondary object.

    Returns
    -------
    Mc : float
        Chirp mass of the binary system.
    eta : float
        Symmetric mass ratio of the binary system.
    '''
    eta = eta_of_q(m1/m2)
    return Mc_of_M_eta(m1+m2,eta), eta


#-----spin functions-----
def chi_s(chi1,chi2):
    '''
    Calculate the symmetric spin parameter (chi_s) given the individual spins (chi1,chi2).

    Parameters
    ----------
    chi1 : float
        The spin of the primary object.
    chi2 : float
        The spin of the secondary object.

    Returns
    -------
    chi_s : float
        The symmetric spin parameter.
    '''
    return 0.5*(chi1+chi2)

def chi_a(chi1,chi2):
    '''
    Calculate the antisymmetric spin parameter (chi_a) given the individual spins (chi1,chi2).

    Parameters
    ----------
    chi1 : float
        The spin of the primary object.
    chi2 : float
        The spin of the secondary object.

    Returns
    -------
    chi_a : float
        The antisymmetric spin parameter.
    '''
    return 0.5*(chi1-chi2)

def chi_eff(m1,m2,chi1,chi2):
    '''
    Calculate the effective spin parameter (chi_eff) given the individual spins (chi1,chi2)
    and component masses (m1,m2).

    Parameters
    ----------
    m1 : float
        Mass of the primary object.
    m2 : float
        Mass of the secondary object.
    chi1 : float
        The spin of the primary object.
    chi2 : float
        The spin of the secondary object.

    Returns
    -------
    chi_eff : float
        The effective spin parameter.
    '''
    return (m1 * chi1 + m2 * chi2) / (m1+m2)


#-----distance parameter functions-----
def DL_of_z_ap_cosmo(z, cosmo=None):
    '''
    Calculate the luminosity distance (DL) given the redshift (z) using astropy.

    Parameters
    ----------
    z : float
        The redshift.
    cosmo : astropy.cosmology, optional
        The cosmology. Default is Planck18.

    Returns
    -------
    DL : float
        The luminosity distance.
    '''
    if cosmo is None: cosmo = ap.cosmology.Planck18
    return cosmo.luminosity_distance(z).value

def z_of_DL_ap_cosmo(DL, cosmo=None):
    '''
    Calculate the redshift (z) given the luminosity distance (DL) using astropy.

    Parameters
    ----------
    DL : float
        The luminosity distance.
    cosmo : astropy.cosmology, optional
        The cosmology. Default is Planck18.

    Returns
    -------
    z : float
        The redshift.
    '''
    if cosmo is None: cosmo = ap.cosmology.Planck18
    return ap.cosmology.z_at_value(cosmo.luminosity_distance, DL * ap.units.Mpc).value


#-----tidal parameter functions-----
def lam_t_of_lam_12_eta(lam1,lam2,eta): # from arXiv:1402.5156
    '''
    Calculate the effective tidal parameter (lambda_t)
    given the individual tidal parameters (lambda1, lambda2) and symmetric mass ratio (eta).

    Parameters
    ----------
    lam1 : float
        The tidal parameter of the primary object.
    lam2 : float
        The tidal parameter of the secondary object.
    eta : float
        The symmetric mass ratio of the binary system.

    Returns
    -------
    lam_t : float
        The effective tidal parameter.
    '''
    return (8./13. * ( (1. + 7. * eta - 31. * eta**2) * (lam1 + lam2) +
                       delta_of_eta(eta) * (1. + 9. * eta - 11. * eta**2) * (lam1 - lam2) ))

def delta_lam_t_of_lam_12_eta(lam1,lam2,eta): # from arXiv:1402.5156
    '''
    Calculate the delta-effective tidal parameter (delta_lambda_t)
    given the individual tidal parameters (lambda1, lambda2) and symmetric mass ratio (eta).

    Parameters
    ----------
    lam1 : float
        The tidal parameter of the primary object.
    lam2 : float
        The tidal parameter of the secondary object.
    eta : float
        The symmetric mass ratio of the binary system.

    Returns
    -------
    delta_lam_t : float
        The delta-effective tidal parameter.
    '''
    return (0.5 * ( delta_of_eta(eta) * (1319. - 13272. * eta + 8944. * eta**2) / 1319. * (lam1 + lam2)+
                          (1319. - 15910. * eta + 32850. * eta**2 + 3380. * eta**3) / 1319. * (lam1 - lam2) ))

def lam_ts_of_lam_12_eta(lam1,lam2,eta): # from arXiv:1402.5156
    '''
    Calculate the effective tidal parameter (lambda_t) and the delta-effective tidal parameter (delta_lambda_t)
    given the individual tidal parameters (lambda1, lambda2) and symmetric mass ratio (eta).

    Parameters
    ----------
    lam1 : float
        The tidal parameter of the primary object.
    lam2 : float
        The tidal parameter of the secondary object.
    eta : float
        The symmetric mass ratio of the binary system.

    Returns
    -------
    lam_t : float
        The effective tidal parameter.
    delta_lam_t : float
        The delta-effective tidal parameter.
    '''
    delta = delta_of_eta(eta)
    lam_t = 8./13. * ( (1. + 7. * eta - 31. * eta**2) * (lam1 + lam2) +
                       delta * (1. + 9. * eta - 11. * eta**2) * (lam1 - lam2) )
    delta_lam_t = 0.5 * ( delta * (1319. - 13272. * eta + 8944. * eta**2) / 1319. * (lam1 + lam2)+
                          (1319. - 15910. * eta + 32850. * eta**2 + 3380. * eta**3) / 1319. * (lam1 - lam2) )
    return lam_t, delta_lam_t

def lam1_of_lam_ts_eta(lam_t, delta_lam_t, eta):
    '''
    Calculate the tidal parameter of the primary object (lambda1)
    given the effective tidal parameter (lambda_t) and the delta-effective tidal parameter (delta_lambda_t).

    Parameters
    ----------
    lam_t : float
        The effective tidal parameter.
    delta_lam_t : float
        The delta-effective tidal parameter.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    lam1 : float
        The tidal parameter of the primary object.
    '''
    delta = delta_of_eta(eta)
    return ((-(-6.76923076923077*delta_lam_t*delta*(-0.09090909090909091 - 0.8181818181818182*eta + 1.*eta**2) +
            19.076923076923077*delta_lam_t*(-0.03225806451612903 - 0.22580645161290322*eta + 1.*eta**2) +
            3.3904473085670963*delta*(0.1474731663685152 - 1.4838998211091234*eta + 1.*eta**2)*lam_t -
            1.281273692191054*(0.39023668639053255 - 4.707100591715976*eta + 9.718934911242604*eta**2 + 1.*eta**3)*lam_t))/
            (8.881784197001252e-16*eta - 1.4210854715202004e-14*eta**2 + 2.842170943040401e-14*eta**3 + 4.500379075056848*eta**4 - 232.4912812736922*eta**5))

def lam2_of_lam_ts_eta(lam_t, delta_lam_t, eta):
    '''
    Calculate the tidal parameter of the secondary object (lambda2)
    given the effective tidal parameter (lambda_t) and the delta-effective tidal parameter (delta_lambda_t).

    Parameters
    ----------
    lam_t : float
        The effective tidal parameter.
    delta_lam_t : float
        The delta-effective tidal parameter.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    lam2 : float
        The tidal parameter of the secondary object.
    '''
    delta = delta_of_eta(eta)
    return ((delta_lam_t*(-1.5296267736621122e-19 + 3.0592535473242243e-19*delta*eta + 7.342208513578138e-18*eta**2 + 9.789611351437518e-18*eta**3 +
            (0.011550173712335778 - 9.789611351437518e-18*delta)*eta**4 + 0.11646425159938675*eta**5) +
            (-3.8240669341552804e-20 + 3.8240669341552804e-20*delta + (-4.588880320986336e-19 - 6.118507094648449e-19*delta)*eta +
            (9.789611351437518e-18 - 1.2237014189296897e-18*delta)*eta**2 + (-9.789611351437518e-18 + 1.4684417027156276e-17*delta)*eta**3 +
            (-0.0014297928149279568 - 0.007954723326344955*delta)*eta**4 + (0.07386363636363634 - 0.005511061254304498*delta)*eta**5)*lam_t)/
            (eta*(0.0909090909090909 - 0.09090909090909091*delta + (0.6363636363636364 - 0.8181818181818181*delta)*eta +
            (-2.818181818181818 + 1.*delta)*eta**2)*(-3.82026549483612e-18 + 6.112424791737792e-17*eta - 1.2224849583475584e-16*eta**2 -
            0.01935719503287065*eta**3 + 1.*eta**4)))

def lam_12_of_lam_ts_eta(lam_t,delta_lam_t,eta):
    '''
    Calculate the individual tidal parameters (lambda1, lambda2)
    given the effective tidal parameter (lambda_t) and the delta-effective tidal parameter (delta_lambda_t).

    Parameters
    ----------
    lam_t : float
        The effective tidal parameter.
    delta_lam_t : float
        The delta-effective tidal parameter.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    lam1 : float
        The tidal parameter of the primary object.
    lam2 : float
        The tidal parameter of the secondary object.
    '''
    delta = delta_of_eta(eta)
    lam1 = ((-(-6.76923076923077*delta_lam_t*delta*(-0.09090909090909091 - 0.8181818181818182*eta + 1.*eta**2) +
            19.076923076923077*delta_lam_t*(-0.03225806451612903 - 0.22580645161290322*eta + 1.*eta**2) +
            3.3904473085670963*delta*(0.1474731663685152 - 1.4838998211091234*eta + 1.*eta**2)*lam_t -
            1.281273692191054*(0.39023668639053255 - 4.707100591715976*eta + 9.718934911242604*eta**2 + 1.*eta**3)*lam_t))/
            (8.881784197001252e-16*eta - 1.4210854715202004e-14*eta**2 + 2.842170943040401e-14*eta**3 + 4.500379075056848*eta**4 - 232.4912812736922*eta**5))
    lam2 = ((delta_lam_t*(-1.5296267736621122e-19 + 3.0592535473242243e-19*delta*eta + 7.342208513578138e-18*eta**2 + 9.789611351437518e-18*eta**3 +
            (0.011550173712335778 - 9.789611351437518e-18*delta)*eta**4 + 0.11646425159938675*eta**5) +
            (-3.8240669341552804e-20 + 3.8240669341552804e-20*delta + (-4.588880320986336e-19 - 6.118507094648449e-19*delta)*eta +
            (9.789611351437518e-18 - 1.2237014189296897e-18*delta)*eta**2 + (-9.789611351437518e-18 + 1.4684417027156276e-17*delta)*eta**3 +
            (-0.0014297928149279568 - 0.007954723326344955*delta)*eta**4 + (0.07386363636363634 - 0.005511061254304498*delta)*eta**5)*lam_t)/
            (eta*(0.0909090909090909 - 0.09090909090909091*delta + (0.6363636363636364 - 0.8181818181818181*delta)*eta +
            (-2.818181818181818 + 1.*delta)*eta**2)*(-3.82026549483612e-18 + 6.112424791737792e-17*eta - 1.2224849583475584e-16*eta**2 -
            0.01935719503287065*eta**3 + 1.*eta**4)))
    return lam1, lam2


#-----derivatives of spin and mass functions-----
def del_Mc_M_of_eta(eta):
    '''
    Calculate the partial derivative of the total mass (M) with respect to the chirp mass (Mc) given the symmetric mass ratio (eta).

    Parameters
    ----------
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    float
        The partial derivative of the total mass with respect to the chirp mass.
    '''
    return eta**-0.6

def del_eta_M_of_Mc_eta(Mc,eta):
    '''
    Calculate the partial derivative of the total mass (M) with respect to the symmetric mass ratio (eta)
    given the chirp mass (Mc) and the symmetric mass ratio (eta).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    float
        The partial derivative of the symmetric mass ratio with respect to the total mass.
    '''
    return -3./5. * Mc * eta**-1.6

def del_Mc_m1_of_Mc_eta(Mc,eta):
    '''
    Calculate the partial derivative of the primary mass (m1) with respect to the chirp mass (Mc)
    given the chirp mass (Mc) and the symmetric mass ratio (eta).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    float
        The partial derivative of the primary mass with respect to the chirp mass.
    '''
    delta = delta_of_eta(eta)
    return 1./2 * del_Mc_M_of_eta(eta) * (1 + delta)

def del_eta_m1_of_Mc_eta(Mc,eta):
    '''
    Calculate the partial derivative of the primary mass (m1) with respect to the symmetric mass ratio (eta)
    given the chirp mass (Mc) and the symmetric mass ratio (eta).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    float
        The partial derivative of the primary mass with respect to the symmetric mass ratio.
    '''
    M = M_of_Mc_eta(Mc,eta)
    delta = delta_of_eta(eta)
    return 1./2 * del_eta_M_of_Mc_eta(Mc,eta) * (1 + delta) - M/delta

def del_Mc_m2_of_Mc_eta(Mc,eta):
    '''
    Calculate the partial derivative of the secondary mass (m2) with respect to the chirp mass (Mc)
    given the chirp mass (Mc) and the symmetric mass ratio (eta).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    float
        The partial derivative of the secondary mass with respect to the chirp mass.
    '''
    delta = delta_of_eta(eta)
    return 1./2 * del_Mc_M_of_eta(eta) * (1 - delta)

def del_eta_m2_of_Mc_eta(Mc,eta):
    '''
    Calculate the partial derivative of the secondary mass (m2) with respect to the symmetric mass ratio (eta)
    given the chirp mass (Mc) and the symmetric mass ratio (eta).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.

    Returns
    -------
    float
        The partial derivative of the secondary mass with respect to the symmetric mass ratio.
    '''
    M = M_of_Mc_eta(Mc,eta)
    delta = delta_of_eta(eta)
    return 1./2 * del_eta_M_of_Mc_eta(Mc,eta) * (1 - delta) + M/delta

def del_Mc_chi_eff(Mc,eta,chi1,chi2):
    '''
    Calculate the partial derivative of the effective spin parameter (chi_eff) with respect to the chirp mass (Mc)
    given the chirp mass (Mc), symmetric mass ratio (eta), and individual spins (chi1,chi2).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.
    chi1 : float
        The spin of the primary object.
    chi2 : float
        The spin of the secondary object.

    Returns
    -------
    float
        The partial derivative of the effective spin parameter with respect to the chirp mass.
    '''
    M = M_of_Mc_eta(Mc,eta)
    m1, m2 = m1_m2_of_M_eta(M,eta)
    return -1./M * del_Mc_M_of_eta(eta) * chi_eff(m1,m2,chi1,chi2) + 1./M * (del_Mc_m1_of_Mc_eta(Mc,eta) * chi1 + del_Mc_m2_of_Mc_eta(Mc,eta) * chi2)

def del_eta_chi_eff(Mc,eta,chi1,chi2):
    '''
    Calculate the partial derivative of the effective spin parameter (chi_eff) with respect to the symmetric mass ratio (eta)
    given the chirp mass (Mc), symmetric mass ratio (eta), and individual spins (chi1,chi2).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.
    chi1 : float
        The spin of the primary object.
    chi2 : float
        The spin of the secondary object.

    Returns
    -------
    float
        The partial derivative of the effective spin parameter with respect to the symmetric mass ratio.
    '''
    M = M_of_Mc_eta(Mc,eta)
    m1, m2 = m1_m2_of_M_eta(M,eta)
    return -1./M * del_eta_M_of_Mc_eta(Mc,eta) * chi_eff(m1,m2,chi1,chi2) + 1./M * (del_eta_m1_of_Mc_eta(Mc,eta) * chi1 + del_eta_m2_of_Mc_eta(Mc,eta) * chi2)

def del_chi1_chi_eff(Mc,eta,chi1,chi2):
    '''
    Calculate the partial derivative of the effective spin parameter (chi_eff) with respect to the spin of the primary object (chi1)
    given the chirp mass (Mc), symmetric mass ratio (eta), and individual spins (chi1,chi2).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.
    chi1 : float
        The spin of the primary object.
    chi2 : float
        The spin of the secondary object.

    Returns
    -------
    float
        The partial derivative of the effective spin parameter with respect to the spin of the primary object.
    '''
    M = M_of_Mc_eta(Mc,eta)
    m1, m2 = m1_m2_of_M_eta(M,eta)
    return 1./M * (m1 + m2 * chi2)

def del_chi2_chi_eff(Mc,eta,chi1,chi2):
    '''
    Calculate the partial derivative of the effective spin parameter (chi_eff) with respect to the spin of the secondary object (chi2)
    given the chirp mass (Mc), symmetric mass ratio (eta), and individual spins (chi1,chi2).

    Parameters
    ----------
    Mc : float
        The chirp mass.
    eta : float
        The symmetric mass ratio.
    chi1 : float
        The spin of the primary object.
    chi2 : float
        The spin of the secondary object.

    Returns
    -------
    float
        The partial derivative of the effective spin parameter with respect to the spin of the secondary object.
    '''
    M = M_of_Mc_eta(Mc,eta)
    m1, m2 = m1_m2_of_M_eta(M,eta)
    return 1./M * (m2 + m1 * chi1)
