import numpy as np
import copy

class Genome( object ):

    def __init__( self, tree, weight_matrix,  expression_matrix, joint_vector):
        """Creates a genome

        tree : Tree
        weight_matrix : n x ( n - num_sensors ) float matrix
        expression_matrix : n x ( n - num_sensors ) bool matrix
        joint_vector : num_joints float vector
        """
        self.tree = copy.deepcopy( tree )
        self.weight_matrix = np.copy( weight_matrix )
        self.expression_matrix = np.copy( expression_matrix )
        self.joint_vector = np.copy( joint_vector )

        self.num_neurons, _ = np.shape( self.weight_matrix )


        # create neuron to branch map
        self.num_branches = self.tree.get_num_branches()
        self.num_sensors = self.tree.get_num_branches()

        self.num_motors = self.num_branches
        self.num_hidden = self.num_neurons - self.num_sensors - self.num_motors
        self.hidden_per_branch = self.num_hidden // self.num_branches

        # create neuron to branch map
        # maps each neuron to a specific branch id
        self.neuron_to_branch_map = np.zeros( self.num_neurons )

        index = 0
        for i in range( self.num_branches ):
            for j in range( self.hidden_per_branch ):
                self.neuron_to_branch_map[index] = i
                index += 1

        for i in range( self.num_branches ):
            self.neuron_to_branch_map[index] = i
            index += 1

        leaves = self.tree.get_leaves()
        for leaf in leaves:
            self.neuron_to_branch_map[index] = leaf.branch_id
            index += 1

    @classmethod
    def random( cls,
                tree,
                sparsity,
                hidden_per_branch = 3,
                weight_range = [-1, 1],
                joint_range = [0, 1] ):
        """Create a random genome"""
        num_branches = tree.get_num_branches()
        num_sensors = tree.get_num_leaves()
        num_motors = num_branches
        num_hidden = hidden_per_branch * num_branches
        num_neurons = num_hidden + num_motors + num_sensors

        m = num_neurons
        n = num_neurons - num_sensors

        weight_matrix = np.random.random( size = ( m, n ) ) * ( weight_range[1] - weight_range[0] ) + weight_range[0]
        expression_matrix = np.random.choice( [0, 1], size = ( m, n ), p = [ sparsity, 1 - sparsity ] )
        joint_vector = np.random.random( size = num_motors ) * ( joint_range[1] - joint_range[0] ) + joint_range[0]

        return cls( tree, weight_matrix, expression_matrix, joint_vector )

    def send_to_simulator( self, sim, write_back = True ):
        """Sends genome to simulator"""

        branches = self.tree.get_branches()

        # send bodies
        body_ids = []
        joint_ids = []
        motor_ids = []
        sensor_ids = []

        h_neuron_ids = []
        m_neuron_ids = []
        s_neuron_ids = []

        sensors = { 'ray' : [], 'angle' : [], 'position' : [] }

        for i, branch in enumerate( branches ):
            # send cylinder body
            alpha = 0.5 + ( i / ( len( branches ) - 1 ) ) / 2
            # print( alpha )
            body_id = sim.send_cylinder( position = branch.position,
                                         orientation = branch.direction,
                                         length = branch.length )
                                         # color = ( 1, alpha, alpha ) )

            # send position sensors and save ids
            x_pos = sim.send_position_x_sensor( body_id, write_back = write_back )
            y_pos = sim.send_position_y_sensor( body_id, write_back = write_back )
            sensors[ 'position' ].append( ( x_pos, y_pos ) )

            # send joints
            if i == 0:
                parent = -1
            else:
                parent = body_ids[ branch.parent.branch_id ]

            # get joint range from joint_vector
            joint_spread = abs( branch.joint_range * self.joint_vector[i] )

            joint_id = sim.send_hinge_joint( parent, body_id,
                                             axis = ( 0, 0, 1 ),
                                             anchor = branch.base,
                                             joint_range = ( -joint_spread, +joint_spread ) )

            angle = sim.send_proprioceptive_sensor( joint_id, write_back = write_back )
            sensors[ 'angle' ].append( angle )

            # send motor
            motor_id = sim.send_rotary_actuator( joint_id )
            for _ in range( self.hidden_per_branch ): # 3 hidden neurons per branch
                h_neuron = sim.send_hidden_neuron()
                h_neuron_ids.append( h_neuron )

            m_neuron = sim.send_motor_neuron( motor_id )
            m_neuron_ids.append( m_neuron )
            # if leaf send ray and sensor
            if branch.is_leaf:
                ray = sim.send_ray( body_id,
                                    position = branch.tip,
                                    direction = branch.direction )

                ray_sensor = sim.send_ray_sensor( ray, write_back = write_back )
                sensors[ 'ray' ].append( ray_sensor )

                s_neuron = sim.send_sensor_neuron( ray_sensor )

                sensor_ids.append( ray_sensor )
                s_neuron_ids.append( s_neuron )


            body_ids.append( body_id )
            joint_ids.append( joint_id )
            motor_ids.append( motor_id )

        # send synapses
        source_neuron_ids = h_neuron_ids + m_neuron_ids + s_neuron_ids
        target_neuron_ids = h_neuron_ids + m_neuron_ids

        for i,source in enumerate( source_neuron_ids ):
            for j,target in enumerate( target_neuron_ids ):
                if self.expression_matrix[i, j] == 1:
                    # print( self.weight_matrix[i, j])
                    sim.send_synapse( source, target, self.weight_matrix[i, j] )
                    # print( i, j, source, target, self.weight_matrix[i, j] )

        if write_back:
            return sensors

    def calc_joint_cost( self, cost_vector = None ):
        if cost_vector is None:
            cost_vector = np.ones_like( self.joint_vector )

        max_cost = np.sum( cost_vector )

        # calcs joint cost as the average value of the joints
        return np.sum( self.joint_vector * cost_vector ) / max_cost

    def calc_connection_cost( self, cost_matrix = None ):
        if cost_matrix is None:
            cost_matrix = np.ones_like( self.expression_matrix )
        
        # calcs joint cost based on cost matrix weighting of each connection
        max_cost = np.sum( cost_matrix )

        return np.sum( cost_matrix * self.expression_matrix ) / max_cost


class Tree( object ):
    def __init__( self,
                  base,
                  current_depth,
                  max_depth,
                  direction,
                  child_angle,
                  joint_range,
                  length,
                  decay,
                  branch_id = 0 ):

        # set paramaters
        self.base = base[:]
        self.direction = direction[:]
        self.length = length
        self.tip = self.base + self.direction * length
        self.position = self.base + self.direction * length / 2.0
        self.joint_range = joint_range
        self.depth = current_depth
        self.max_depth = max_depth

        self.n_branches = 2**( max_depth + 1 - current_depth ) - 1

        if current_depth == 0:
            self.is_root = True
            self.branch_id = 0
            self.parent = None
        else:
            self.is_root = False
            self.branch_id = branch_id

        self.last_id = self.branch_id

        self.children = []
        if current_depth == max_depth:
            self.is_leaf = True
            return
        else:
            self.is_leaf = False

        # create children
        for i in range( 2 ):
            theta = ( -1 ) ** ( 1 + i % 2 ) * child_angle

            child_direction = np.array( [ direction[0] * np.cos( theta ) + direction[1] * np.sin( theta ),
                                         -direction[0] * np.sin( theta ) + direction[1] * np.cos( theta ),
                                         0 ] )


            child_tree = Tree( base = self.tip[:],
                                         current_depth = current_depth +1,
                                         max_depth = max_depth,
                                         child_angle = child_angle * decay,
                                         direction = child_direction,
                                         joint_range = joint_range * decay,
                                         length = length,
                                         decay = decay,
                                         branch_id = self.last_id + 1 )
            self.last_id = child_tree.last_id
            child_tree.parent = self
            self.children.append( child_tree )

    def get_branches( self ):
        """Performs depth first listify of branches. i.e. in id order"""

        branches = [ self ]

        for i in range( len( self.children ) ):
            branches.extend( self.children[i].get_branches() )

        return branches

    def get_branch( self, target_id ):
        """Gets branch with branch_id = target_id"""
        if ( target_id == self.branch_id ):
            return self

        if ( self.is_leaf ):
            return None

        if ( target_id < self.children[1].branch_id ):
            return self.children[0].get_branch( target_id )
        else:
            return self.children[1].get_branch( target_id )

    def calc_path_length( self, branch_id_1, branch_id_2 ):
        # find common ancestor

        if ( branch_id_1 == branch_id_2 ):
            return 0
        
        branch_1 = self.get_branch( branch_id_1 )
        branch_2 = self.get_branch( branch_id_2 )

        branch_1_path = []
        branch_2_path = []

        # get path back to root from the first branch
        while ( branch_1 != None ):
            branch_1_path.append( branch_1.branch_id )
            branch_1 = branch_1.parent

        # get path back to root from the second branch
        while ( branch_2 != None ):
            branch_2_path.append( branch_2.branch_id )
            branch_2 = branch_2.parent

        # get length to common ancestor and return
        for id1 in branch_1_path:
            if id1 in branch_2_path:
                index1 = branch_1_path.index( id1 )
                index2 = branch_2_path.index( id1 )
                
                return ( index1 + index2 )

    def get_num_branches( self ):
        return 2 ** ( self.max_depth + 1 ) - 1

    def get_num_leaves( self ):
        return self.get_num_branches() - ( 2 ** ( self.max_depth ) - 1 )

    def get_leaves( self ):
        """Gets the leaf objects of the tree"""
        branches = self.get_branches()
        leaves = []

        for branch in branches:
            if branch.is_leaf:
                leaves.append( branch )

        return leaves

def send_environment( sim, env_str, tree,
                      distance_dict = { '0' : 2, '1' : 4 },
                      color_dict = { '0' : ( 0, 0, 0 ), '1' : ( 1, 1 ,1 ) },
                      size_dict = { '0' : 0.5, '1' : 0.5 } ):
    """Sends a 'retina' environment to the simulator"""

    leaves = tree.get_leaves()
    assert( len( env_str ) == len( leaves ) ), ( 'Number of leaves much match number of objects' )

    seen_sensors = []
    for env_char, leaf in zip( env_str, leaves ):
        pos = leaf.tip + distance_dict[env_char] * leaf.direction # location is n units away from leaf tip

        # send object
        body = sim.send_cylinder( position = pos,
                           length = pos[2] * 2.0,
                           radius = size_dict[env_char],
                           capped = False,
                           color = color_dict[env_char] )
        sensor = sim.send_is_seen_sensor( body )
        seen_sensors.append( sensor )

    return seen_sensors


if __name__ == "__main__":
    import sys
    sys.path.insert( 0, '../../Code/pyrosim' )

    import pyrosim
    import matplotlib.pyplot as plt

    np.random.seed( 0 )
    tree = Tree( base = np.array( [ 0, 0, 0.5 ] ),
                 current_depth = 0,
                 max_depth = 2,
                 direction = np.array( [ 0, 1, 0 ] ),
                 child_angle = np.pi / 4.0,
                 joint_range = np.pi / 2.0,
                 length = 0.5,
                 decay = 0.5 )

    g = Genome.random( tree, sparsity = 0.5, hidden_per_branch = 2 )
    # g.expression_matrix[:, :] = 1
    # g.weight_matrix[:, :] = -1
    # g.joint_vector[:] = 1.0
    # g.joint_vector[:] = 0.1
    # print( g.expression_matrix )
    # print( g.joint_vector )
    # # leaves = tree.get_leaves()

    sim = pyrosim.Simulator( eval_steps = -1, draw_shadows = False, play_paused = True, use_textures = False )

    sensors = g.send_to_simulator( sim )


    sim.set_camera( xyz = ( 0, 2, 8 ), hpr = ( 90, -90, 0 ) )
    sim.start()
    sim.wait_to_finish()

    # print( sensors )
    # print( sim._raw_cerr )

    # print( sim.get_sensor_data() )