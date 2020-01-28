#include "GameState.h"

using namespace Eigen;

//TODO add asserts bruv
GameState::GameState(std::shared_ptr<Game> game, int action, std::shared_ptr<GameState> parent){
	this->isExpanded = false;;
	this->parent = parent;
	this->game = game;
	this->action = action;
	this->childW = ArrayXf::Zero(game->getActionSize());
	this->childP = ArrayXf::Zero(game->getActionSize());
	this->childN = ArrayXf::Zero(game->getActionSize());
	this->children = std::vector<std::shared_ptr<GameState>>(game->getActionSize());  
}

void GameState::addVirtualLoss(){
	std::shared_ptr<GameState> current = shared_from_this();

	while (current->getParent()){
		current->updateW(1);
		current = current->getParent();
	}
}

void GameState::removeVirtualLoss(){
	std::shared_ptr<GameState> current = shared_from_this();

	while (current->getParent()){
		current->updateW(-1);
		current = current->getParent();
	}
}

std::shared_ptr<GameState> GameState::getParent(){
	return this->parent;
} 

std::shared_ptr<GameState> GameState::select(int cpuct){
	std::shared_ptr<GameState> current = shared_from_this();
	int action;
	ArrayXf puct;//, q, u;
	
	while (current->isExpanded and not current->game->ended()){
		/*q = current->childQ();
		u = current->childU(cpuct);
		*/
		puct = current->childQ() + current->childU(cpuct);
		action = getBestAction(puct, current);
		
		/*std::cout << "puct" << puct<< std::endl;
		std::cout << "puct q" << q<< std::endl;
		std::cout << "puct u" << u<< std::endl;
		std::cout << "puct action" << action<< std::endl;*/
		
		current = current->play(action);
	}

	return current;
}

int GameState::getBestAction(ArrayXf puct, std::shared_ptr<GameState> current){
	ArrayXf poss = current->game->getPossibleActions();
	float max = std::numeric_limits<float>::lowest();
	int action = -1;
	for (int i; i < poss.size(); i++){
		if (poss[i] != 0){
			if(puct[i] > max){
				action = i; 	
				max = puct[i];	
			}	
		}				
	}
				
	return action;			
}

std::shared_ptr<GameState> GameState::play(int action){
	if (!this->children[action]){
		std::shared_ptr<Game> t = std::move(this->game->copy());
		t->play(action);

		this->children[action] = std::make_shared<GameState>(t, action, shared_from_this()); 
	}

	return this->children[action];
}

void GameState::backup(float v){
	std::shared_ptr<GameState> current = shared_from_this();
	while (current->getParent()){
		current->incN();
		current->updateW(v);
		current = current->getParent();
		v *= -1;
	}
}

ArrayXf GameState::getchildW(){
	return this->childW;
}

void GameState::updateW(float v){
	this->parent->childW[this->action] += v;
}

void GameState::incN(){
	this->parent->childN[this->action]++;
}

float GameState::getN(){
	return this->parent->childN(this->action);
}

float GameState::getP(){
	return this->parent->childP(this->action);
}

ArrayXf GameState::childQ(){
	return this->childW / (ArrayXf::Ones(game->getActionSize()) + this->childN);
}

ArrayXf GameState::childU(float cpuct){
	return cpuct 
			* this->childP 
			* ( 
				std::sqrt(this->getN())
				/
				(ArrayXf::Ones(game->getActionSize()) + this->childN)
			   );
}

ArrayXf GameState::dirichlet_distribution(ArrayXf alpha){
	ArrayXf res =  ArrayXf::Zero(alpha.size());
	std::random_device rd;
	std::mt19937 gen(rd());

	for(int i; i < alpha.size(); i++){
		std::gamma_distribution<double> dist(alpha(i),1);
		auto sample = dist(gen); 
		res(i) = sample;
	}
	
	if (res.sum() == 0){
		return res; //TODO: hacky 
	} else {
		return res/res.sum();
	}	
} 


void GameState::expand(ArrayXf p, float dirichlet_alpha){
	this->isExpanded = true;
	
	ArrayXf poss = this->game->getPossibleActions();

	if (!this->parent->parent){
		ArrayXf alpha = ArrayXf::Ones(p.size())*dirichlet_alpha;
		ArrayXf d = dirichlet_distribution(alpha);
		p = 0.75*p + 0.25*d;
	}

	this->childP = this->getValidActions(p, poss);

	//std::cout << "p inside " << p << std::endl;
	//std::cout << "poss " << poss << std::endl;
	//std::cout << "childp" << this->childP << std::endl;
}

ArrayXf GameState::getValidActions(ArrayXf pred, ArrayXf poss){
	ArrayXf valid_actions = (pred * poss);

	if (not valid_actions.any()){
		valid_actions = poss;
	}

	return valid_actions/valid_actions.sum();
}

ArrayXf GameState::getSearchPolicy(float temp){
	/*std::cout << "W" << std::endl;
	std::cout << childW << std::endl;
	std::cout << "N" << std::endl;
	std::cout << childN << std::endl;
	std::cout << "P" << std::endl;
	std::cout << childP << std::endl;*/
	ArrayXf count = pow(this->childN,1/temp);
	//std::cout << "childN " << this->childN << std::endl;
	//std::cout << "count " << count << std::endl;
	//std::cout << "count.sum " << count.sum()<< std::endl;
	return count/count.sum();
}

bool GameState::endGame(){
	return game->ended();
}

bool GameState::getPlayer(){
	return game->getPlayer();
}
bool GameState::getWinner(){
	return game->getWinner();
}

MatrixXf GameState::getCanonicalBoard(){
	return this->game->getBoard()*this->game->getPlayer();
}

//Todo: this shouldnt belong to this class
std::vector<MatrixXf> GameState::getNetworkInput(){
	std::vector<MatrixXf> game_state ;
	auto dims = this->game->getBoardSize();
	std::shared_ptr<GameState> current = shared_from_this();
	game_state.insert(game_state.begin(), current->getCanonicalBoard());

	for(int i = 0; i < this->game->getInputPlanes() - 1; i++){
		if (current->getParent()){
			current =  current->getParent();
			game_state.insert(game_state.begin(), current->getCanonicalBoard());
		}
		else{
			game_state.insert(game_state.begin(), MatrixXf::Zero(dims[0], dims[1]));
		}
	
	}
	return game_state;
}