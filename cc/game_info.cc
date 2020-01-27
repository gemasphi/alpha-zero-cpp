#include <Game.h>
#include <json.hpp>
#include <iostream>

using json = nlohmann::json;

int main(int argc, char** argv){
	std::unique_ptr<Game> d = Game::create(argv[1]);
	json j; 
	j["input_planes"] = d->getInputPlanes();
	j["output_planes"] = d->getOutputPlanes();
	j["board_size"] = d->getBoardSize();
	j["action_size"] = d->getActionSize();
	std::cout << j;
	return 0;
}
