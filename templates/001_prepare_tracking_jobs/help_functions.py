import numpy as np
import xline
import xtrack

def from_normal_to_physical_space(df, egeom_1, egeom_2, ptau_max, W=None, invW=None):

    init_canonical_6D = np.empty([1, 6])
    init_canonical_6D[0,0] = 0.
    init_canonical_6D[0,1] = 0.
    init_canonical_6D[0,2] = 0.
    init_canonical_6D[0,3] = 0.
    init_canonical_6D[0,4] = 0.
    init_canonical_6D[0,5] = ptau_max
    init_normalized_6D_temp = np.tensordot(invW, init_canonical_6D, [1,1]).T

    r_list = df['normalized amplitude in xy-plane'].values
    theta_list = df['angle in xy-plane [deg]'].values*np.pi/180 # [rad]

    A1_A2_in_sigma = np.array((r_list*np.cos(theta_list),r_list*np.sin(theta_list)))
    n_particles = len(A1_A2_in_sigma[0])

    A1_A2_in_sigma_unrolled = A1_A2_in_sigma.reshape(n_particles, 2)

    init_normalized_6D = np.empty([n_particles, 6])
    init_normalized_6D[:,0] = A1_A2_in_sigma_unrolled[:,0] * np.sqrt(egeom_1)
    init_normalized_6D[:,1] = 0.
    init_normalized_6D[:,2] = A1_A2_in_sigma_unrolled[:,1] * np.sqrt(egeom_2)
    init_normalized_6D[:,3] = 0.
    init_normalized_6D[:,4] = 0.
    init_normalized_6D[:,5] = init_normalized_6D_temp[0,5]

    init_canonical_coordinates = np.tensordot(W, init_normalized_6D, [1,1]).T

    return init_canonical_coordinates, A1_A2_in_sigma, n_particles


def get_particles_distribution(egeom_1, egeom_2, ptau_max, r_list, theta_list, W=None, invW=None):

    init_canonical_6D = np.empty([1, 6])
    init_canonical_6D[0,0] = 0.
    init_canonical_6D[0,1] = 0.
    init_canonical_6D[0,2] = 0.
    init_canonical_6D[0,3] = 0.
    init_canonical_6D[0,4] = 0.
    init_canonical_6D[0,5] = ptau_max
    init_normalized_6D_temp = np.tensordot(invW, init_canonical_6D, [1,1]).T


    A1_A2_in_sigma = np.array([[(r*np.cos(theta),r*np.sin(theta)) for r in r_list] for theta in theta_list])
    n_particles = len(theta_list)*len(r_list)

    A1_A2_in_sigma_unrolled = A1_A2_in_sigma.reshape(n_particles, 2)


    init_normalized_6D = np.empty([n_particles, 6])
    init_normalized_6D[:,0] = A1_A2_in_sigma_unrolled[:,0] * np.sqrt(egeom_1)
    init_normalized_6D[:,1] = 0.
    init_normalized_6D[:,2] = A1_A2_in_sigma_unrolled[:,1] * np.sqrt(egeom_2)
    init_normalized_6D[:,3] = 0.
    init_normalized_6D[:,4] = 0.
    init_normalized_6D[:,5] = init_normalized_6D_temp[0,5]

    init_canonical_coordinates = np.tensordot(W, init_normalized_6D, [1,1]).T

    return init_canonical_coordinates, A1_A2_in_sigma, n_particles

def get_DA_distribution(n_sigma, ptau_max, egeom_1, egeom_2, r_N=1144, theta_N=11, W=None, invW=None):

    init_canonical_6D = np.empty([1, 6])
    init_canonical_6D[0,0] = 0.
    init_canonical_6D[0,1] = 0.
    init_canonical_6D[0,2] = 0.
    init_canonical_6D[0,3] = 0.
    init_canonical_6D[0,4] = 0.
    init_canonical_6D[0,5] = ptau_max
    init_normalized_6D_temp = np.tensordot(invW, init_canonical_6D, [1,1]).T

    deg2rad = np.pi/180.
    deg_step = 90./(theta_N+1)
    theta_min = deg_step * deg2rad
    theta_max = (90. - deg_step) * deg2rad
    r_min = 0.
    r_max = n_sigma

    A1_A2_in_sigma = np.array([[(r*np.cos(theta),r*np.sin(theta)) for r in np.linspace(r_min,r_max,r_N)] for theta in np.linspace(theta_min,theta_max,theta_N)])

    n_particles = theta_N*r_N

    A1_A2_in_sigma_unrolled = A1_A2_in_sigma.reshape(n_particles, 2)


    init_normalized_6D = np.empty([n_particles, 6])
    init_normalized_6D[:,0] = A1_A2_in_sigma_unrolled[:,0] * np.sqrt(egeom_1)
    init_normalized_6D[:,1] = 0.
    init_normalized_6D[:,2] = A1_A2_in_sigma_unrolled[:,1] * np.sqrt(egeom_2)
    init_normalized_6D[:,3] = 0.
    init_normalized_6D[:,4] = 0.
    init_normalized_6D[:,5] = init_normalized_6D_temp[0,5]

    init_canonical_coordinates = np.tensordot(W, init_normalized_6D, [1,1]).T

    return init_canonical_coordinates, A1_A2_in_sigma, n_particles

def add_to_closed_orbit(init_canonical_6D, partCO, particle_id=None):
    pp = partCO.copy()
    pp.x += init_canonical_6D[:,0]
    pp.px += init_canonical_6D[:,1]
    pp.y += init_canonical_6D[:,2]
    pp.py += init_canonical_6D[:,3]
    pp.zeta += init_canonical_6D[:,4]
    pp.delta += init_canonical_6D[:,5]
    if particle_id is None:
         pp.particle_id = np.arange(len(pp.x), dtype=np.int64)
    else:
         pp.particle_id = np.int64(particle_id)
    return pp

def apply_closed_orbit(init_canonical_coordinates, partCO):
    for ii in range(6):
        init_canonical_coordinates[:,ii] += partCO[ii]

    return init_canonical_coordinates

def J1_J2_from_canonical(canonical_coordinates, invW, partCO):
    coords = canonical_coordinates.copy()
    for ii in range(6):
        coords[:,ii] -= partCO[ii]

    normalized_coords = np.tensordot(invW, coords, [1,1]).T
    J1 = 0.5*(normalized_coords[:,0]**2 + normalized_coords[:,1]**2)
    J2 = 0.5*(normalized_coords[:,2]**2 + normalized_coords[:,3]**2)

    return J1, J2

#def get_xtrack_particle_set(init_canonical_coordinates, p0c_eV):
#    n_part = init_canonical_coordinates.shape[0]
#
#    ps = xtrack.ParticlesSet()
#    p = ps.Particles(num_particles=n_part)
#
#    for i_part in range(n_part):
#        part = pysixtrack.Particles(p0c=p0c_eV)
#
#        part.x    = init_canonical_coordinates[i_part, 0]
#        part.px   = init_canonical_coordinates[i_part, 1]
#        part.y    = init_canonical_coordinates[i_part, 2]
#        part.py   = init_canonical_coordinates[i_part, 3]
#        part.tau  = init_canonical_coordinates[i_part, 4]
#        part.ptau = init_canonical_coordinates[i_part, 5]
#
#        part.particle_id = i_part
#        part.state  = 1
#        part.elemid = 0
#        part.turn   = 0
#        
#        p.from_pysixtrack(part, i_part)
#
#    return p

def get_xtrack_particle_set(context, init_canonical_coordinates, p0c_eV):
    particles = xtrack.Particles(_context=context,
            p0c  = p0c_eV,
            x    = init_canonical_coordinates[:, 0],
            px   = init_canonical_coordinates[:, 1],
            y    = init_canonical_coordinates[:, 2],
            py   = init_canonical_coordinates[:, 3],
            tau  = init_canonical_coordinates[:, 4],
            ptau = init_canonical_coordinates[:, 5],
    )
    return particles
