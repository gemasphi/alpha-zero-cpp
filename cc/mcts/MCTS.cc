#include "MCTS.h"

using namespace Eigen;

ArrayXf MCTS::simulate(std::shared_ptr<Game> game, NNWrapper& model, MCTS::Config cfg){		
	return cfg.parallel ? do_parallel_simulate(std::make_shared<GameState>(game), model, cfg) 
						: do_simulate(std::make_shared<GameState>(game), model, cfg);  
}

ArrayXf MCTS::simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg){		
	return cfg.parallel ? do_parallel_simulate(root, model, cfg) 
						: do_simulate(root, model, cfg);  
}


ArrayXf MCTS::simulate_random(std::shared_ptr<Game> game, MCTS::Config cfg){
	std::shared_ptr<GameState> root = std::make_shared<GameState>(game);
	std::shared_ptr<GameState> leaf; 
	for (int i = 0; i < cfg.n_simulations + 1; i++){
		leaf = root->select(cfg.cpuct);

		if (leaf->endGame()){
			leaf->backup(leaf->getWinner()*root->getPlayer());
			continue;
		}

		int value = leaf->rollout();
		leaf->expand(ArrayXf::Ones(root->game->getActionSize())/root->game->getActionSize(), cfg.dirichlet_alpha);
		leaf->backup(value);

		tree_to_dot(root);
	}

	return root->getSearchPolicy(cfg.temp);
}


ArrayXf MCTS::do_simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg){
	std::shared_ptr<GameState> leaf; 

	for (int i = 0; i < cfg.n_simulations + 1; i++){
		leaf = root->select(cfg.cpuct);

		if (leaf->endGame()){
			leaf->backup(leaf->getWinner()*root->getPlayer());
			continue;
		}

		NN::Output res = model.maybeEvaluate(leaf);
		
		leaf->expand(res.policy, cfg.dirichlet_alpha);
		leaf->backup(res.value);

	}

	return root->getSearchPolicy(cfg.temp);
}


ArrayXf MCTS::do_parallel_simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg){
	int simulations_to_run = cfg.n_simulations / omp_get_max_threads(); 
	//we do more than the n_simulations currently
	
	for(int i = 0; i < simulations_to_run + 1; i++){
		std::vector<std::shared_ptr<GameState>> leafs;
		
		#pragma omp parallel
		{
			std::shared_ptr<GameState> leaf = root->select(cfg.cpuct);
			if (leaf->endGame()){
				leaf->backup(leaf->getWinner());
			}
			else{
				leaf->addVirtualLoss(cfg.vloss);
				
				#pragma omp critical
				leafs.push_back(leaf);
			}
		}
		if (leafs.size() == 0){
			break;
		}
		
		std::vector<NN::Output> res = model.maybeEvaluate(leafs);

		#pragma omp parallel for
		for(unsigned int i = 0; i < leafs.size(); i++){
			leafs[i]->removeVirtualLoss(cfg.vloss);
			leafs[i]->expand(res[i].policy, cfg.dirichlet_alpha);
			leafs[i]->backup(res[i].value);
		}

	}

	return root->getSearchPolicy(cfg.temp);
}
	


void MCTS::tree_to_dot_aux(std::shared_ptr<GameState> root, std::stringstream& dot){
	for (auto &child : root->children){
		if (child){
			dot <<"    \""<< *root <<"\" -> \""<< *child <<"\";\n"; 
			tree_to_dot_aux(child, dot);
		}
	}
} 

void MCTS::tree_to_dot(std::shared_ptr<GameState> root){
	std::stringstream dot;
	dot << "digraph BST {\n";
    dot << "    node [fontname=\"Arial\"];\n";

	tree_to_dot_aux(root, dot);

    dot << "}\n";

    std::cout << dot.str();
}


