#include "NNObserver.h"

using namespace Eigen;


void setScheduling(std::thread &th, int policy, int priority) {
	sched_param sch_params;
    sch_params.sched_priority = priority;
    if(pthread_setschedparam(th.native_handle(), policy, &sch_params)) {
        std::cerr << "Failed to set Thread scheduling : " << std::strerror(errno) << std::endl;
    }
}

NNObserver::NNObserver(NNWrapper& nnwrapper, std::string filename, bool watchFile) :
					  nnwrapper(nnwrapper), watchFile(watchFile){
	if (watchFile){
		this->fileWatcher = std::thread(&NNObserver::setup_inotify, this, filename);
		setScheduling(this->fileWatcher, SCHED_BATCH, 0);
	}
}
	

NNObserver::~NNObserver(){
	if (this->watchFile){
		close(this->inotifyFD);
		this->fileWatcher.join();
	}
}

//todo: error handling
void NNObserver::setup_inotify(std::string file){
	this->inotifyFD = inotify_init();
	inotify_add_watch(this->inotifyFD, file.c_str(), IN_ATTRIB | IN_MODIFY | IN_CREATE | IN_DELETE );

	int buff_size = ( 1024 * ( sizeof (struct inotify_event) + NAME_MAX + 1) );
	char buffer[ buff_size ];
	
	fd_set watch_set;
    FD_ZERO( &watch_set );
    FD_SET( this->inotifyFD, &watch_set );

	std::cout << "Watching for updates" << std::endl;
	while(fcntl(this->inotifyFD, F_GETFD) != -1) {
		struct timeval tv = {120, 0};   
		if(select( this->inotifyFD+1, &watch_set, NULL, NULL, &tv) == 1){
			std::unique_lock lock(*(this->nnwrapper.getModelMutex()));
			std::cout << "New model detected" << std::endl;
			int length = read( this->inotifyFD, buffer, buff_size ); 
			this->nnwrapper.load(file);
			tv = {120, 0};   
			lock.unlock();
	    }
	}
}

