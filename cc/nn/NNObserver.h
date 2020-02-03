#ifndef NNOBSERVER_H     
#define  NNOBSERVER_H

#include <iostream>
#include "NNUtils.h"
#include "NNWrapper.h"

#include <sys/inotify.h>
#include <fcntl.h>

#include <mutex>  
#include <shared_mutex>
#include <thread>
#include <pthread.h>

class NNWrapper;   //forward declaration

class NNObserver{
	private:
		NNWrapper& nnwrapper;
		bool watchFile;
		
		std::thread fileWatcher;
		int inotifyFD; 

		void setup_inotify(std::string directory);

	public:
		NNObserver(NNWrapper& nnwrapper, std::string filename, bool watchFile);
		~NNObserver(); 
};

#endif 