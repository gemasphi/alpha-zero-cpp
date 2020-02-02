#include "NNWrapper.h"

using namespace Eigen;

#define EVENT_SIZE          ( sizeof (struct inotify_event) )
#define EVENT_BUF_LEN       ( 1024 * ( EVENT_SIZE + NAME_MAX + 1) )

//todo: error handling
void NNWrapper::setup_inotify(std::string file){
	int fd = inotify_init();
	int wd = inotify_add_watch(fd, file.c_str(), IN_ATTRIB | IN_MODIFY | IN_CREATE | IN_DELETE );

	char buffer[ EVENT_BUF_LEN ];
	
	fd_set watch_set;
    FD_ZERO( &watch_set );
    FD_SET( fd, &watch_set );

	while(fcntl(fd, F_GETFD) != -1) {
		//if (terminateFileWatcher) return;
		
		std::cout << "wacthcing" << std::endl;
		if(select( fd+1, &watch_set, NULL, NULL, NULL ) == 1){
			int length = read( fd, buffer, EVENT_BUF_LEN ); 
			this->load(file);
	    }
	}
	
	//close (fd); 
	//std::cout << "close" << std::endl;
}


NN::Output NNWrapper::maybeEvaluate(std::shared_ptr<GameState> leaf){
	std::stringstream ss;
	ss << leaf->getCanonicalBoard();
	std::string board = ss.str();	
	

	if (this->inCache(board)) {
		return this->getCachedEl(board);
	}
	else{
		NN::Output res = this->predict(NN::Input(leaf->getNetworkInput()))[0];
		
		//pragma omp critical(netCache)
		this->inserInCache(board, res);
		
		return res;
	}

}

std::vector<NN::Output> NNWrapper::maybeEvaluate(std::vector<std::shared_ptr<GameState>> leafs){
	std::vector<std::vector<MatrixXf>> _boards;

	for (auto leaf: leafs){
		_boards.push_back(leaf->getNetworkInput());
	}	

	std::vector<NN::Output> res 
		= this->predict(NN::Input(_boards));

	return res;
}

NNWrapper::NNWrapper(std::string filename){
	omp_init_lock(&this->modelock);

	this->fileWatcher = std::thread(&NNWrapper::setup_inotify, this, filename);
	this->load(filename);
}
	


bool NNWrapper::inCache(std::string board){
	return this->netCache.find(board) != this->netCache.end();
}

NN::Output NNWrapper::getCachedEl(std::string board){
	return this->netCache.at(board);
}

void NNWrapper::inserInCache(std::string board, NN::Output o){
	this->netCache.insert({board, o});
}

void NNWrapper::load(std::string filename){

	try {
		std::cout << "loading the model\n";
		omp_set_lock(&(this->modelock));
			this->module = torch::jit::load(filename, torch::kCPU);
		omp_unset_lock(&(this->modelock));
		
		netCache = std::unordered_map<std::string, NN::Output>();
		std::cout << "model loaded\n";
	}
	catch (const c10::Error& e) {
		std::cout << "error reloading the model, using old model\n";
	}

}

std::vector<NN::Output> NNWrapper::predict(NN::Input input){
	std::vector<torch::jit::IValue> jit_inputs;
	jit_inputs.push_back(input.boards.to(at::kCPU));
	
	//omp_set_lock(&(this->modelock));
	torch::jit::IValue output = this->module.forward(jit_inputs);
	//omp_unset_lock(&(this->modelock));
	
	std::vector<NN::Output> o;

	for(int i = 0; i < input.batch_size; i++){
		o.push_back(NN::Output(output, i));
	} 

	return o;
}
