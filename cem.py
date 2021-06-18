import numpy as np

def cem(F,num_iters,N,d,sigma=0.1,alpha=0.001):

	fixed_w = np.array([0.1, -0.3, 0.5])
	mu = np.random.randn(d)

	sigma = 0.1
	alpha = 0.001 # learning rate

	for i in range(num_iters):
    	# print current fitness of the most likely parameter setting
    	if i % 20 == 0:
        	print('iter %d. w: %s, solution: %s, reward: %f' % 
        	(i, str(mu), str(fixed_w), F(fixed_w,mu)))
    	noise = np.random.normal(size=(N,d))
    	x_guesses = mu + sigma*noise
    	Fs = F(fixed_w, x_guesses)

    	# normalize scores
    	# X = X - mu
    	Fs_mean = np.mean(Fs)
    	Fs = Fs - Fs_mean
    	# compute sample std dev
    	std_dev = np.sqrt(np.mean(Fs**2))
    	Fs = Fs/std_dev
    	
    	"""
    	# where each row N[j] is weighted by A[j]
    	w = w + alpha/(npop*sigma) * np.dot(N.T, A)
    	"""
    
    	# compute update
    	res1 = np.dot(noise.T,Fs)
    	mu += 1/(sigma*N)*alpha * res1

    return mu
