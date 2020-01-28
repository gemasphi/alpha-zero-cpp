#include "MCTS.h"

using namespace Eigen;

MCTS::MCTS(float cpuct, float dirichlet_alpha){
	this->cpuct = cpuct;
	this->dirichlet_alpha = dirichlet_alpha;
}

ArrayXf MCTS::simulate(std::shared_ptr<Game> game, NNWrapper& model, float temp, int n_simulations){
	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(game, 0, fakeparentparent);
	std::shared_ptr<GameState> root = std::make_shared<GameState>(game, 0, fakeparent);

	std::shared_ptr<GameState> leaf; 

	for (int i = 0; i < n_simulations; i++){
		leaf = root->select(this->cpuct);
		

		if (leaf->endGame()){
			leaf->backup(leaf->getWinner()*root->getPlayer());
			continue;
		}

		NN::Output res = model.maybeEvaluate(leaf);
		
		leaf->expand(res.policy, dirichlet_alpha);
		leaf->backup(res.value);
	}
	return root->getSearchPolicy(temp);

}

/*
ArrayXf threaded_simulate(std::shared_ptr<GameState> root, NNWrapper& model, float temp = 1, int n_simulations = 5){
	int eval_size = 4;
	int simulations_to_run = n_simulations / eval_size; 
	std::vector<GameState> leafs;
	//we do more than the n_simulations currently
	
	for(int i = 0; i < 5 + 1; i++){
		#pragma omp parallel
		{
			GameState leaf = root->select(this->cpuct);

			if (leaf->endGame()){
				leaf->backup(leaf->getWinner());
				continue;
			}

			leaf->addVirtualLoss();
			leafs.push_back(leaf);
		}

		if (leafs.size == 0){
			break;
		}
		
		NN::Output output = model.predict(leafs); 
			#pragma omp for
			{
				for (NN::Output o& : output){
				leaf.remove_virtual_loss();
				leaf.expand(o.p, dirichlet_alpha);
				leaf.backup(o.v);
				}
		}
	}
	return root->getSearchPolicy(temp);
}
	*/



