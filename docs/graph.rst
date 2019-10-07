Graph Cration
=============

Graphs
------

.. py:class:: Cvnn

.. py:method:: create_mlp_graph(self, shape)

	Creates a complex-fully-connected dense graph using a shape as parameter

        :param shape: List of tuple
            1. each number of shape[i][0] correspond to the total neurons of layer i.
            2. a string in shape[i][1] corresponds to the activation function listed on TODO
                ATTENTION: shape[0][0] will be ignored! A future version will apply the activation function to the input but not implemented for the moment.
            Where i = 0 corresponds to the input layer and the last value of the list corresponds to the output layer.
        :return: None

Others
------

.. py:method:: restore_graph_from_meta(self, latest_file=None)
	
	Restores an existing graph from meta data file

        :param latest_file: Path to the file to be restored. If no latest_file given and self.automatic_restore is True, the function will try to load the newest metadata inside `saved_models/` folder.
        :return: None

