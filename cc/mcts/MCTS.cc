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
	return simulate_random(std::make_shared<GameState>(game),cfg);
}
ArrayXf MCTS::simulate_random(std::shared_ptr<GameState> root, MCTS::Config cfg){
	std::shared_ptr<GameState> leaf; 

	for (int i = 0; i < cfg.n_simulations + 1; i++){
		leaf = root->select(cfg.cpuct);

		if (leaf->endGame()){
			leaf->backup(leaf->getWinner()*leaf->parent->getPlayer());
			continue;
		}

		int value = leaf->rollout();
		leaf->expand(ArrayXf::Ones(root->game->getActionSize())/root->game->getActionSize(), cfg.dirichlet_alpha);
		leaf->backup(value*leaf->parent->getPlayer());

	}

//	tree_to_dot(root);
	return root->getSearchPolicy(cfg.temp);
}


ArrayXf MCTS::do_simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg){
	std::shared_ptr<GameState> leaf; 

	for (int i = 0; i < cfg.n_simulations + 1; i++){
		leaf = root->select(cfg.cpuct);

		if (leaf->endGame()){
			leaf->backup(leaf->getWinner()*leaf->parent->getPlayer());
			continue;
		}

		NN::Output res = model.maybeEvaluate(leaf);
		
		leaf->expand(res.policy, cfg.dirichlet_alpha);
		leaf->backup(res.value);

	}

	return root->getSearchPolicy(cfg.temp);
}


ArrayXf MCTS::do_parallel_simulate(std::shared_ptr<GameState> root, NNWrapper& model, MCTS::Config cfg){
	int simulations_to_run = cfg.n_simulations / cfg.batchSize; 
	
	omp_set_dynamic(0);
	for(int j = 0; j < simulations_to_run + 1; j++){
		std::vector<std::shared_ptr<GameState>> leafs;
		
		#pragma omp parallel for num_threads(cfg.n_threads)
		for(int k = 0; k < cfg.batchSize; k++){
			std::shared_ptr<GameState> leaf = root->select(cfg.cpuct);
			if (leaf->endGame()){
				leaf->backup(leaf->getWinner()*leaf->parent->getPlayer());
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

		
		std::future<std::vector<NN::Output>> res_future = model.maybeEvaluate(leafs, cfg.globalBatchSize);
		std::vector<NN::Output> res = res_future.get(); 

		#pragma omp parallel for num_threads(cfg.n_threads)
		for(unsigned int i = 0; i < leafs.size(); i++){
			leafs[i]->removeVirtualLoss(cfg.vloss);
			leafs[i]->expand(res[i].policy, cfg.dirichlet_alpha);
			leafs[i]->backup(res[i].value);
		}

	}

	//std::cout <<"simulation" <<std::endl;

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


