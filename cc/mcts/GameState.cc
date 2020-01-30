#include "GameState.h"
#include <assert.h> 

using namespace Eigen;

//TODO add asserts bruv
GameState::GameState(std::shared_ptr<Game> game, int action, std::shared_ptr<GameState> parent){
	this->parent = parent;
	this->children = std::vector<std::shared_ptr<GameState>>(game->getActionSize());  
	this->action = action;
	this->childW = ArrayXf::Zero(game->getActionSize());
	this->childP = ArrayXf::Zero(game->getActionSize());
	this->childN = ArrayXf::Zero(game->getActionSize());
	this->game = game;
}

GameState::GameState(std::shared_ptr<Game> game){
	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(game, 0, fakeparentparent);
	this->parent = fakeparent;
	this->action = 0;
	this->children = std::vector<std::shared_ptr<GameState>>(game->getActionSize());  
	this->childW = ArrayXf::Zero(game->getActionSize());
	this->childP = ArrayXf::Zero(game->getActionSize());
	this->childN = ArrayXf::Zero(game->getActionSize());
	this->game = game;
}

std::ostream& operator<<(std::ostream& os, const GameState& gs)
{
    os  << "Child N\n" << gs.childN  << std::endl
    	<< "Child Q\n" << gs.childP  << std::endl
    	<< "Child W\n" << gs.childW  << std::endl
    	<< "Action" << gs.action 
    	<< "Board\n" << gs.game->getBoard() << std::endl;
    
    return os;
}

std::shared_ptr<GameState> GameState::select(float cpuct){
	std::shared_ptr<GameState> current = shared_from_this();
	int action;
	
	while (current->isExpanded and not current->game->ended()){
		action = current->getBestAction(cpuct);
		current = current->play(action);
	}

	return current;
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
}

void GameState::backup(float v){
	std::shared_ptr<GameState> current = shared_from_this();
	while (current->parent){
		current->incN();
		current->updateW(v);
		current = current->parent;
		v *= -1;
	}
}

void GameState::addVirtualLoss(){
	std::shared_ptr<GameState> current = shared_from_this();

	while (current->parent){
		current->updateW(1);
		current = current->parent;
	}
}

void GameState::removeVirtualLoss(){
	std::shared_ptr<GameState> current = shared_from_this();

	while (current->parent){
		current->updateW(-1);
		current = current->parent;
	}
}


int GameState::getBestAction(float cpuct){
	ArrayXf puct = this->childQ() + this->childU(cpuct);
	ArrayXf poss = this->game->getPossibleActions();
	
	float max = std::numeric_limits<float>::lowest();
	int action = -1;

	for (int i = 0; i < poss.size(); i++){
		if (poss[i] != 0){
			if(puct[i] > max){
				action = i; 	
				max = puct[i];	
			}	
		}				
	}
	
	assert(action != -1);			
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



void GameState::updateW(float v){
	this->parent->childW[this->action] += v;
}

void GameState::incN(){
	this->parent->childN[this->action]++;
}

float GameState::getN(){
	return this->parent->childN(this->action);
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


ArrayXf GameState::getValidActions(ArrayXf pred, ArrayXf poss){
	ArrayXf valid_actions = (pred * poss);

	if (not valid_actions.any()){
		valid_actions = poss;
	}

	return valid_actions/valid_actions.sum();
}

ArrayXf GameState::getSearchPolicy(float temp){
	ArrayXf count = pow(this->childN,1/temp);
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

std::vector<MatrixXf> GameState::getNetworkInput(){
	std::vector<MatrixXf> game_state ;
	auto dims = this->game->getBoardSize();
	std::shared_ptr<GameState> current = shared_from_this();
	game_state.insert(game_state.begin(), current->getCanonicalBoard());

	for(int i = 0; i < this->game->getInputPlanes() - 1; i++){
		if (current->parent){
			current =  current->parent;
			game_state.insert(game_state.begin(), current->getCanonicalBoard());
		}
		else{
			game_state.insert(game_state.begin(), MatrixXf::Zero(dims[0], dims[1]));
		}
	
	}
	return game_state;
}


std::shared_ptr<GameState> GameState::getChild(int action){
	std::shared_ptr<GameState> fakeparentparent;
	std::shared_ptr<GameState> fakeparent = std::make_shared<GameState>(this->game, 0, fakeparentparent); 	
	
	std::shared_ptr<GameState> child = this->play(action);
	child->parent = fakeparent; 

	return child;
}