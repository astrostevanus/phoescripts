#Mostly taken from http://secondearths.sakura.ne.jp/exojax/, I just made some functions to 'summarise' them for my personal work
import scipy.interpolate as sci
import jax.numpy as jnp
import numpy as np
from jax import vmap, jit
import h5py

Tref=296.0

dir_heliosk='/mnt/phoenest/stevanus/PlanetSpecGen/HELIOS-K/data/'
dir_kurucz=str(dir_heliosk)+"kuruczCor/"
dir_exomol=str(dir_heliosk)+"exomol/"
dir_hitemp=str(dir_heliosk)+"hitemp/"

""""
Binary file format
Kurucz 'd,d,d,d,d'

HITEMP '<4c,d,d,d,d,d,d,d,d' skipcol=1
    from heliosk script hitran.cpp
    fwrite(&cid, 4*sizeof(char), 1, binFile);
    fwrite(&nu, sizeof(double), 1, binFile);
    fwrite(&S, sizeof(double), 1, binFile);
    fwrite(&EL, sizeof(double), 1, binFile);
    fwrite(&A, sizeof(double), 1, binFile);
    fwrite(&delta, sizeof(double), 1, binFile);
    fwrite(&gammaAir, sizeof(double), 1, binFile);
    fwrite(&gammaSelf, sizeof(double), 1, binFile);
    fwrite(&n, sizeof(double), 1, binFile);

Exomol 'd,d,d,d'
    from heliosk script prepareExomol.cpp
    fwrite(&nu, sizeof(double), 1, OutFile);
    fwrite(&S, sizeof(double), 1, OutFile);
    fwrite(&EL, sizeof(double), 1, OutFile);
    fwrite(&A, sizeof(double), 1, OutFile);
"""
elt0=np.array([[  1, "H"  , 1.00794],
        [  2, "He" , 4.002602],
        [  3, "Li" , 6.941],
        [  4, "Be" , 9.012182],
        [  5, "B"  , 10.811],
        [  6, "C"  , 12.011],
        [  7, "N"  , 14.00674],
        [  8, "O"  , 15.9994],
        [  9, "F"  , 18.9984032],
        [ 10, "Ne" , 20.1797],
        [ 11, "Na" , 22.989768],
        [ 12, "Mg" , 24.3050],
        [ 13, "Al" , 26.981539],
        [ 14, "Si" , 28.0855],
        [ 15, "P"  , 30.973762],
        [ 16, "S"  , 32.066],
        [ 17, "Cl" , 35.4527],
        [ 18, "Ar" , 39.948],
        [ 19, "K"  , 39.0983],
        [ 20, "Ca" , 40.078],
        [ 21, "Sc" , 44.955910],
        [ 22, "Ti" , 47.88],
        [ 23, "V"  , 50.9415],
        [ 24, "Cr" , 51.9961],
        [ 25, "Mn" , 54.93805],
        [ 26, "Fe" , 55.847],
        [ 27, "Co" , 58.93320],
        [ 28, "Ni" , 58.6934],
        [ 29, "Cu" , 63.546],
        [ 30, "Zn" , 65.39],
        [ 31, "Ga" , 69.723],
        [ 32, "Ge" , 72.61],
        [ 33, "As" , 74.92159],
        [ 34, "Se" , 78.96],
        [ 35, "Br" , 79.904],
        [ 36, "Kr" , 83.80],
        [ 37, "Rb" , 85.4678],
        [ 38, "Sr" , 87.62],
        [ 39, "Y"  , 88.90585],
        [ 40, "Zr" , 91.224],
        [ 41, "Nb" , 92.90638],
        [ 42, "Mo" , 95.94],
        [ 43, "Tc" , 97.9072],
        [ 44, "Ru" ,101.07],
        [ 45, "Rh" ,102.90550],
        [ 46, "Pd" ,106.42],
        [ 47, "Ag" ,107.8682],
        [ 48, "Cd" ,112.411],
        [ 49, "In" ,114.818],
        [ 50, "Sn" ,118.710],
        [ 51, "Sb" ,121.757],
        [ 52, "Te" ,127.60],
        [ 53, "I"  ,126.90447],
        [ 54, "Xe" ,131.29],
        [ 55, "Cs" ,132.90543],
        [ 56, "Ba" ,137.327],
        [ 57, "La" ,138.9055],
        [ 58, "Ce" ,140.115],
        [ 59, "Pr" ,140.90765],
        [ 60, "Nd" ,144.24],
        [ 61, "Pm" ,144.9127],
        [ 62, "Sm" ,150.36],
        [ 63, "Eu" ,151.965],
        [ 64, "Gd" ,157.25],
        [ 65, "Tb" ,158.92534],
        [ 66, "Dy" ,162.50],
        [ 67, "Ho" ,164.93032],
        [ 68, "Er" ,167.26],
        [ 69, "Tm" ,168.93421],
        [ 70, "Yb" ,173.04],
        [ 71, "Lu" ,174.967],
        [ 72, "Hf" ,178.49],
        [ 73, "Ta" ,180.9479],
        [ 74, "W"  ,183.84],
        [ 75, "Re" ,186.207],
        [ 76, "Os" ,190.23],
        [ 77, "Ir" ,192.22],
        [ 78, "Pt" ,195.08],
        [ 79, "Au" ,196.96654],
        [ 80, "Hg" ,200.59],
        [ 81, "Tl" ,204.3833],
        [ 82, "Pb" ,207.2],
        [ 83, "Bi" ,208.98037],
        [ 84, "Po" ,208.9824],
        [ 85, "At" ,209.9871],
        [ 86, "Rn" ,222.0176],
        [ 87, "Fr" ,223.0197],
        [ 88, "Ra" ,226.0254],
        [ 89, "Ac" ,227.0278],
        [ 90, "Th" ,232.0381],
        [ 91, "Pa" ,231.03588],
        [ 92, "U"  ,238.0289],
        [ 93, "Np" ,237.0482],
        [ 94, "Pu" ,244.0642],
        [ 95, "Am" ,243.0614],
        [ 96, "Cu" ,247.0703],
        [ 97, "Bk" ,247.0703],
        [ 98, "Cf" ,251.0796],
        [ 99, "Es" ,252.0830],
        [100, "Fm" ,257.0951],
        [101, "Md" ,258.0984],
        [102, "No" ,259.1011],
        [103, "Lr" ,262.1098],
        [104, "Rf" ,261.1089],
        [105, "Db" ,262.1144],
        [106, "Sg" ,263.1186],
        [107, "Bh" ,264.12],
        [108, "Hs" ,265.1306],
        [109, "Mt" ,268.00],
        [110, "Ds" ,268.00],
        [111, "Rg" ,272.00],
        [112, "Cn" ,277.00],
        [113, "Uut" ,0.00],
        [114, "Fl" ,289.00],
        [115, "Uup" ,0.00],
        [116, "Lv" ,289.00],
        [117, "Uus" ,294.00],
        [118, "Uuo" ,293.00]])

def dplist_calc(plist):
    dlogP=(np.log10(plist)[-1]-np.log10(plist)[0])/(len(plist)-1.)
    k=10**-dlogP
    dplist = (1.0-k)*plist
    return dlogP, k, dplist

def Qt_interpol(pf_file):
    """
    Interpolating partition function vs temperature

    Args:
        pf_file: partition function file
    Return:
        Interpolator of partition function as a function of temperature
    """
    T_hit,QT_hit=np.loadtxt(pf_file,unpack=True)
    Qt_interp=sci.interp1d(T_hit,QT_hit)
    return (Qt_interp)

def qt_qt0_barklem_collet_heliosk(atomic_name, iion, Tarr):
    """
    Calculating the Q(Tarr)/Q(Tref) for atomic species using Barklem Collet .pf files in HELIOS-K database

    Args:
        atomic_name: the name of the atomic species
        iion       : ionisation level (e.g., neutral=1, singly ionized=2, etc.)
        Tarr       : array of temperatures
    Return:
        array of partition function ratio as a function of temperatures
    """
    #Finding the Kurucz filename for specific atomic species and ionisation level
    atomic_number=int(elt0[:,0][elt0[:,1]==atomic_name])
    if atomic_number<10:
        filename= "gfnew0"+str(atomic_number)+"0"+str(iion-1)
    else:
        filename= "gfnew"+str(atomic_number)+"0"+str(iion-1)

    #Load Barklem Collet partition files
    T,QT=np.loadtxt(str(dir_kurucz)+str(filename)+".pf",unpack=True)

    #Interpolating partition function vs temperature
    qt_func= sci.interp1d(T,QT)
    return qt_func(Tarr)/qt_func(Tref)

def read_ciah5(filename,nus,nue):
    """READ HITRAN CIA data

    Args:
       filename: HITRAN CIA file name (_2011.cia)
       nus: wavenumber min (cm-1)
       nue: wavenumber max (cm-1)

    Returns:
       nucia: wavenumber (cm-1)
       tcia: temperature (K)
       ac: cia coefficient

    """
    with h5py.File(filename,'r') as h5f_xs_cia:
        nu=h5f_xs_cia['wavenumber'][:]
        data=h5f_xs_cia['ac'][:]
        tcia=h5f_xs_cia['T (K)'][:]

    tcia=np.array(tcia)
    nu=np.array(nu)
    ijnu=np.digitize([nus,nue],nu)
    nucia=np.array(nu[ijnu[0]:ijnu[1]+1])
    ac=data[:,ijnu[0]:ijnu[1]+1]
    return nucia,tcia,ac

def read_binary_file_heliosk(filename, skipcol, dtype):
    """
    Reading HELIOS-K-formated binary file

    Args:
        filename: the name of the binary file
        skipcol : which column to skip
        dtype   : the type of the data
    Return:
        array of data
    """
    with open(filename, "rb") as f:
        numpy_data = np.fromfile(f,dtype)
        data=np.empty((len(numpy_data),len(dtype)-skipcol),dtype="float64")
        for i in range(len(numpy_data)):
            for j in range(skipcol,len(dtype)):
                data[i][j-skipcol]=numpy_data[i][j]
    return data

def load_mol_bin_heliosk(binary_filename, pf_file, molmass, nus, whichdata):
    """
    Loading HITEMP or EXOMOL binary files in HELIOS-K format

    Args:
        binary_filename: name of the HITEMP linelist binary file
        pf_file        : partition function filename
        molmass        : mass of the considered molecule
        nus            : wavenumber (cm-1)
        whichdata      : "exomol" or "hitemp"
    Return:
        if whichdata=="hitemp" or whichdata=="HITEMP":
            an array consists of line positions, line strength at Tref, lower energy,
            Einstein coefficient (A), delta, gamma Air, gamma self and n air within wavenumber range
        elif whichdata=="exomol" or whichdata=="EXOMOL":
            an array consists of line positions, line strength at Tref, lower energy,
            Einstein coefficient (A) within wavenumber range
    """
    from exojax.exojax_wrapper import Qt_interpol
    
    
    if whichdata=="hitemp" or whichdata=="HITEMP":
        Qt_interp=Qt_interpol(str(dir_hitemp)+str(pf_file))
        data=read_binary_file_heliosk(str(dir_hitemp)+str(binary_filename),1,np.dtype('<4c,d,d,d,d,d,d,d,d'))

        mask_nu=(data[:,0]>=nus[0])*(data[:,0]<=nus[-1])
        nu_lines= data[:,0][mask_nu]
        logSij0= np.log(np.array(data[:,1][mask_nu]*molmass/Qt_interp(296.)/6.0221412927e23,dtype="float64"))
        ELower= jnp.array(data[:,2][mask_nu])
        A= jnp.array(data[:,3][mask_nu])
        delta= jnp.array(data[:,4][mask_nu])
        gammaAir=jnp.array(data[:,5][mask_nu])
        gammaSelf= jnp.array(data[:,6][mask_nu])
        n_air= jnp.array(data[:,7][mask_nu])

        return np.array([nu_lines, logSij0, ELower, A, delta, gammaAir, gammaSelf, n_air])

    elif whichdata=="exomol" or whichdata=="EXOMOL":
        from exojax.utils.constants import hcperk
        Qt_interp=Qt_interpol(str(dir_exomol)+str(pf_file))
        QTref=Qt_interp(Tref)
        data=read_binary_file_heliosk(str(dir_exomol)+binary_filename,0,np.dtype('d,d,d,d'))

        mask_nu=(data[:,0]>=nus[0])*(data[:,0]<=nus[-1])
        nu_lines= data[:,0][mask_nu]
        S       = data[:,1][mask_nu]
        ELower  = jnp.array(data[:,2][mask_nu])
        A       = jnp.array(data[:,3][mask_nu])
        logSij0 = np.log(-np.exp(np.array(-hcperk*ELower/Tref),dtype="float64")*np.expm1(-hcperk*nu_lines/Tref)\
                            *molmass/QTref*S/6.0221412927e23)
        
        return np.array([nu_lines, logSij0, ELower, A])
    else:
        print ("Currently only support Exomol and HITEMP linelists")

def load_kurucz_bin_heliosk_exojax(atomic_name, iion, nus):
    """
    Loading kurucz binary files in Exojax specialised format (to include gamma Stark and van Der Waals,
    HELIOS-K format does not include these)

    Args:
        atomic_name: the name of the atomic species
        iion       : ionisation level (e.g., neutral=1, singly ionized=2, etc.)
        nus        : wavenumber (cm-1)
    Return:
        an array consists of line position, log10 line strength at Tref, Einstein coefficient,
        lower energy, upper energy, Gamma Rad, Gamma Stark and Gamma van Der Waals within wavenumber range
    """
    import array
    import os

    atomic_number=int(elt0[:,0][elt0[:,1]==atomic_name])

    if atomic_number<10:
        filename= "gfnew0"+str(atomic_number)+"0"+str(iion-1)
    else:
        filename= "gfnew"+str(atomic_number)+"0"+str(iion-1)

    data_arr = array.array('d')
    with open(str(dir_kurucz)+str(filename)+"_exojax.bin", 'rb') as fin:
        n = os.fstat(fin.fileno()).st_size // 8
        data_arr.fromfile(fin, n)
        data_arr=np.reshape(data_arr,(int(len(data_arr)/8),8))

    wn      = data_arr[:,0] #wavenumber
    logS0   = jnp.array(data_arr[:,1]) #log S0
    A       = jnp.array(np.exp(data_arr[:,2])) # A
    ELow    = jnp.array(data_arr[:,3]) # Elow
    EUp     = jnp.array(data_arr[:,4]) # Eup
    GammaRad= jnp.array(data_arr[:,5]) #log10 GammaRad
    GammaS  = jnp.array(data_arr[:,6]) #log10 GammaStark
    GammavW = jnp.array(data_arr[:,7]) #log10 Gamma van Der Waals
    mask_nu=(wn>nus[0])*(wn<nus[-1])

    return np.array([wn[mask_nu], logS0[mask_nu], A[mask_nu], ELow[mask_nu], EUp[mask_nu], GammaRad[mask_nu], GammaS[mask_nu], GammavW[mask_nu]])

def ionE_atom(atomic_number,iion):
    """
    Args:
        atomic_number: number of the atom (e.g. Fe = 26)
        iion: ionisation level (e.g., neutral=1, singly ionized=2, etc.)
    Return:
        ionisation energy, atomic mass

    """
    from exojax.spec import atomllapi
    ipccd = atomllapi.load_atomicdata()

    ionE = ipccd[ipccd['ielem']==atomic_number].iat[0, 1]
    ionE2= ipccd[ipccd['ielem']==atomic_number].iat[0, 6]
    ionE = ionE * np.where(iion==1, 1, 0) + ionE2 * np.where(iion==2, 1, 0)

    atomic_mass=ipccd[ipccd['ielem']==atomic_number].iat[0, 5]

    return ionE, atomic_mass

def db_exomol(db_linelist, molmass, nus, Tarr, Parr):
    """
    Calculating the line strength (SijM) and broadening width (gammaLM, sigmaDM), and extracting line positions (nu_lines)
    for Exomol linelist
    Args:
        db_linelist: the database of the linelist (e.g. load_exomol_bin_heliosk(linelist_file, pf_file, molmass, nus))
        molmass    : molecular mass in a.m.u.
        nus        : wavenumber (cm-1)
        Tarr       : array of temperatures in K
        Parr       : array of pressure in bar
    Return:
        line positions, line strengths,  Lorentz's width, Doppler's width
    """

    from exojax.spec import gamma_natural, SijT,doppler_sigma
    from exojax.spec.exomol import gamma_exomol
    
    #Calculating Q(Tarr)/Q(Tref)
    qt_qt0=vmap(db_linelist.qr_interp)(Tarr)

    nu_lines=jnp.array(db_linelist.nu_lines)

    #Line strength at t temperature
    SijM=jit(vmap(SijT,(0,None,None,None,0)))(Tarr, db_linelist.logsij0, nu_lines, db_linelist.elower, qt_qt0)

    #Lorentz width: pressure+natural broadening
    gammaLM= jit(vmap(gamma_exomol,(0,0,None,None)))(Parr, Tarr, db_linelist.n_Texp, db_linelist.alpha_ref)\
            +gamma_natural(db_linelist.A)[None,:]

    #Thermal broadening
    sigmaDM=doppler_sigma(nu_lines[None,:],Tarr[:,None],molmass)

    masknan= ~np.isnan(gammaLM[0])

    return nu_lines[masknan], SijM[:,masknan], gammaLM[:,masknan], sigmaDM[:,masknan]

def db_hitemp(db_linelist, molmass, nus, Tarr, Parr, localid, pf_file):
    """
    Calculating the line strength (SijM) and broadening width (gammaLM, sigmaDM), and extracting line positions (nu_lines)
    for HITEMP linelist

    Args:
        db_linelist: the database of the linelist (e.g. load_hitemp_bin_heliosk(linelist_dir,pf_file,molmass,nus))
        molmass    : molecular mass in a.m.u.
        nus        : wavenumber (cm-1)
        Tarr       : array of temperatures in K
        Parr       : array of pressure in bar
        localid    : local id of molecular isotope in hitemp (1= main isotope)
        pf_file    : partition function filename
    Return:
        line positions, line strengths,  Lorentz's width, Doppler's width
    """

    from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
    from exojax.exojax_wrapper import Qt_interpol

    Tref=296.0
    Qt_interp=Qt_interpol(str(dir_hitemp)+str(pf_file))
    qt_qt0=Qt_interp(Tarr)/Qt_interp(Tref)

    Pself=1
    #masking isotope
    iso_mask=(db_linelist.isoid==localid)

    nu_lines= db_linelist.nu_lines[iso_mask]

    SijM=SijT(Tarr[:,None],db_linelist.logsij0[iso_mask][None,:],db_linelist.nu_lines[iso_mask][None,:],\
             db_linelist.elower[iso_mask][None,:],qt_qt0[:,None])

    gammaLM = 0*gamma_hitran(Parr[:,None],Tarr[:,None], Pself, db_linelist.n_air[iso_mask][None,:],
                              db_linelist.gamma_air[iso_mask][None,:],db_linelist.gamma_self[iso_mask][None,:])\
                            + gamma_natural(db_linelist.A[iso_mask])[None,:]

    sigmaDM= doppler_sigma(nu_lines[None,:],Tarr[:,None],molmass)

    masknan= ~np.isnan(gammaLM[0])

    return nu_lines[masknan], SijM[:,masknan], gammaLM[:,masknan], sigmaDM[:,masknan]

def db_vald(db_linelist, atomic_name, iion, nus, Tarr, Parr, vmr_H, vmr_He, vmr_H2):
    """
    Calculating the line strength (SijM) and broadening width (gammaLM, sigmaDM), and extracting line positions (nu_lines)
    for VALD linelist

    Args:
        db_linelist: the database of the linelist (e.g. moldb.AdbVald(valdlines, nus))
        atomic_name: the name of the atomic species
        iion       : ionisation level (e.g., neutral=1, singly ionized=2, etc.)
        nus        : wavenumber (cm-1)
        Tarr       : array of temperatures in K
        Parr       : array of pressure in bar
        vmr_H      : volume mixing ratio of hydrogen
        vmr_He     : volume mixing ratio of helium
        vmr_H2     : volume mixing ratio of molecular hydrogen

    Return:
        line positions, line strengths,  Lorentz's width, Doppler's width
    """

    from exojax.spec import atomll, SijT, doppler_sigma
    from exojax.exojax_wrapper import ionE_atom, qt_qt0_barklem_collet_heliosk

    qt_qt0=qt_qt0_barklem_collet_heliosk(atomic_name, iion,Tarr)

    atomic_number=int(elt0[:,0][elt0[:,1]==atomic_name])

    maskl= (db_linelist.ielem==atomic_number)*(db_linelist.iion==iion)

    ionE, atomic_mass=ionE_atom(atomic_number, iion)

    PH = Parr* vmr_H
    PHe = Parr* vmr_He
    PHH = Parr* vmr_H2

    nu_lines=db_linelist.nu_lines[maskl]

    SijM=jit(vmap(SijT,(0,None,None,None,0)))\
        (Tarr, db_linelist.logsij0[maskl], nu_lines, db_linelist.elower[maskl], qt_qt0.T)


    gammaLM = jit(vmap(atomll.gamma_vald3,(0,0,0,0,None,None,None,None,None,None,None,None,None,None,None)))\
            (Tarr, PH, PHH, PHe, atomic_number, iion, \
                    nu_lines, db_linelist.elower[maskl], db_linelist.eupper[maskl], atomic_mass, ionE, \
                    db_linelist.gamRad[maskl], db_linelist.gamSta[maskl], db_linelist.vdWdamp[maskl], 1.0)

    sigmaDM=jit(vmap(doppler_sigma,(None,0,None)))\
        (nu_lines, Tarr, atomic_mass)

    masknan= ~np.isnan(gammaLM[0])

    return nu_lines[masknan], SijM[:,masknan], gammaLM[:,masknan], sigmaDM[:,masknan], atomic_mass

def db_mol_heliosk(db_linelist, molmass, nus, Tarr, Parr,  pf_file, whichdata):
    """
    Calculating the line strength (SijM) and broadening width (gammaLM, sigmaDM), and extracting line positions (nu_lines)
    for HITEMP/Exomol linelist from HELIOS-K binary files

    Args:
        db_linelist: the database of the linelist (e.g. load_mol_bin_heliosk(binary_filename, pf_file, molmass, nus, "hitemp"))
        molmass    : molecular mass in a.m.u.
        nus        : wavenumber (cm-1)
        Tarr       : array of temperatures in K
        Parr       : array of pressure in bar
        pf_file    : name of the partition function file
    Return:
        line positions, line strengths,  Lorentz's width, Doppler's width
    """
    
    from exojax.exojax_wrapper import Qt_interpol
        
    if whichdata=="hitemp" or whichdata=="HITEMP":
        Qt_interp=Qt_interpol(str(dir_hitemp)+(pf_file))
        qt_qt0=Qt_interp(Tarr)/Qt_interp(Tref)
        
        from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
            
        nu_lines, logSij0, ELower, A, delta, gammaAir, gammaSelf, n_air= db_linelist
    
        Pself=1
        gammaLM = 0*gamma_hitran(Parr[:,None],Tarr[:,None], Pself, n_air[None,:],
                                  gammaAir[None,:], gammaSelf[None,:])\
                                + gamma_natural(A)[None,:]
    elif whichdata=="exomol" or whichdata=="EXOMOL":
        from exojax.spec import gamma_natural, SijT,doppler_sigma
        from exojax.spec.exomol import gamma_exomol
        
        Qt_interp=Qt_interpol(str(dir_exomol)+(pf_file))
        qt_qt0=Qt_interp(Tarr)/Qt_interp(Tref)

        alpha_ref= 0.07 #Default value of Lorentz half-width
        n_Texp= 0.5 #Default value of temperature exponent

        nu_lines,logSij0,ELower,A= db_linelist

        gammaLM= jit(vmap(gamma_exomol,(0,0,None,None)))(Parr,Tarr, n_Texp, alpha_ref)[:,None]\
                +gamma_natural(A)[None,:]
        
    SijM=jit(vmap(SijT,(0,None,None,None,0)))(Tarr, logSij0, nu_lines, ELower, qt_qt0)
    
    sigmaDM= doppler_sigma(nu_lines[None,:],Tarr[:,None],molmass)

    masknan= ~np.isnan(gammaLM[0])

    return nu_lines[masknan], SijM[:,masknan], gammaLM[:,masknan], sigmaDM[:,masknan]


def db_kurucz(db_linelist, atomic_name, iion, nus, Tarr, Parr, vmr_H, vmr_He, vmr_H2):
    """
    Calculating the line strength (SijM) and broadening width (gammaLM, sigmaDM), and extracting line positions (nu_lines)
    for Kurucz linelist from Exojax-formatted binary files

    Args:
        db_linelist: the database of the linelist (e.g. load_mol_bin_heliosk(binary_filename, pf_file, molmass, nus, "exomol"))
        atomic_name: the name of the atomic species
        iion       : ionisation level (e.g., neutral=1, singly ionized=2, etc.)
        nus        : wavenumber (cm-1)
        Tarr       : array of temperatures in K
        Parr       : array of pressure in bar
        vmr_H      : volume mixing ratio of hydrogen
        vmr_He     : volume mixing ratio of helium
        vmr_H2     : volume mixing ratio of molecular hydrogen

    Return:
        line positions, line strengths,  Lorentz's width, Doppler's width
    """
    from exojax.spec import atomll, SijT, doppler_sigma
    from exojax.exojax_wrapper import ionE_atom, qt_qt0_barklem_collet_heliosk

    qt_qt0=qt_qt0_barklem_collet_heliosk(atomic_name, iion,Tarr)

    nu_lines, logSij0, A, elower, eupper, gammaRad, gammaSta, vdWdamp= db_linelist
    atomic_number=int(elt0[:,0][elt0[:,1]==atomic_name])
    ionE, atomic_mass=ionE_atom(atomic_number, iion)

    PH = Parr* vmr_H
    PHe = Parr* vmr_He
    PHH = Parr* vmr_H2

    SijM=jit(vmap(SijT,(0,None,None,None,0)))(Tarr, logSij0, nu_lines, elower, qt_qt0.T)

    gammaLM = jit(vmap(atomll.gamma_vald3,(0,0,0,0,None,None,None,None,None,None,None,None,None,None,None)))\
            (Tarr, PH, PHH, PHe, atomic_number, 1, nu_lines, elower, eupper,
                    atomic_mass, ionE, gammaRad, gammaSta, vdWdamp, 1.0)

    sigmaDM= jit(vmap(doppler_sigma,(None,0,None)))(nu_lines, Tarr, atomic_mass)

    masknan= ~np.isnan(gammaLM[0])
    return nu_lines[masknan],SijM[:,masknan], gammaLM[:,masknan], sigmaDM[:,masknan], atomic_mass

def xs_modit(nu_lines, SijM, gammaLM, speciesmass, R_nugrid, Tarr, nus):
    """
    Calculating cross-section of particular species,

    Args:
        nu_lines   : line positions (cm-1)
        SijM       : line strength at Tarr
        gammaLM    : Lorentz width
        speciesmass: the mass of the species in a.m.u.
        R_nugrid   : spectral resolution from nugrid
        Tarr       : array of temperatures in K
        nus        : wavenumber (cm-1)
    Return:
        array of cross-section (layers x wavenumber)
    """

    from exojax.spec import normalized_doppler_sigma, modit, initspec

    nsigmaDl= normalized_doppler_sigma(Tarr, speciesmass, R_nugrid)[:,np.newaxis]

    dv_lines= nu_lines/R_nugrid

    ngammaLM= gammaLM/dv_lines[None,:]

    dgm_ngammaL= modit.dgmatrix(ngammaLM,0.2)

    cnu, indexnu, R_modit, pmarray= initspec.init_modit(nu_lines,nus)

    xsm= jnp.abs(modit.xsmatrix(cnu, indexnu, R_modit, pmarray, nsigmaDl, ngammaLM, SijM, nus, dgm_ngammaL))

    return xsm

def vac2air(wv_vac):
    s = 1e4/wv_vac
    n = 1.+0.0000834254+0.02406147/(130.-s**2)+0.00015998/(38.9-s**2)
    wv_air=wv_vac/n
    return wv_air
