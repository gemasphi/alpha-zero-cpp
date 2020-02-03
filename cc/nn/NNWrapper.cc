#include "NNWrapper.h"

using namespace Eigen;


void setScheduling(std::thread &th, int policy, int priority) {
	sched_param sch_params;
    sch_params.sched_priority = priority;
    if(pthread_setschedparam(th.native_handle(), policy, &sch_params)) {
        std::cerr << "Failed to set Thread scheduling : " << std::strerror(errno) << std::endl;
    }
}

NNWrapper::NNWrapper(std::string filename, bool watchFile){
	this->watchFile = watchFile;

	if (watchFile){
		this->fileWatcher = std::thread(&NNWrapper::setup_inotify, this, filename);
		//setScheduling(this->fileWatcher, SCHED_RR, 99);
	}
	
	this->load(filename);
}
	

NNWrapper::~NNWrapper(){
	if (this->watchFile){
		close(this->inotifyFD);
		this->fileWatcher.join();
	}
}

//todo: error handling
void NNWrapper::setup_inotify(std::string file){
	this->inotifyFD = inotify_init();
	inotify_add_watch(this->inotifyFD, file.c_str(), IN_ATTRIB | IN_MODIFY | IN_CREATE | IN_DELETE );

	int buff_size = ( 1024 * ( sizeof (struct inotify_event) + NAME_MAX + 1) );
	char buffer[ buff_size ];
	
	fd_set watch_set;
    FD_ZERO( &watch_set );
    FD_SET( this->inotifyFD, &watch_set );

	while(fcntl(this->inotifyFD, F_GETFD) != -1) {
		struct timeval tv = {5, 0};   
		if(select( this->inotifyFD+1, &watch_set, NULL, NULL, &tv) == 1){
			std::unique_lock lock(this->modelMutex);
			std::cout << "New model detected" << std::endl;
			int length = read( this->inotifyFD, buffer, buff_size ); 
			this->load(file);
			lock.unlock();
	    }
	}

	std::cout<<"dead" << std::endl;
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
		this->module = torch::jit::load(filename, torch::kCPU);
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
	
	std::shared_lock lock(this->modelMutex);
	torch::jit::IValue output = this->module.forward(jit_inputs);
	lock.unlock();

	std::vector<NN::Output> o;

	for(int i = 0; i < input.batch_size; i++){
		o.push_back(NN::Output(output, i));
	} 

	return o;
}
