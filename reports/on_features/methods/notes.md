
# plan:
- launch basic losses with few data (1 mask per image) and overfit 
	- 10 images
	- 100 images
	- 1000 images

# logs:
[x] create dataset (with blur)
	- 99999 images
	- using `python data.py datadir 99999 --reset` - sigma blur is 5
	- takes around 1 hour
[x] launch simplest run
	- using `python engine.py train datadir simplest _devrun` (first with --dev then without)
	- simplest didn't really work
- overfitting didn't work
[x] training with hinge and the big dataset starts doing stuff, however:
	- blurry
	- there's still repetition of color

- i've tried optimizing the output directly to check the losses. 
	- it works on the big square (sample index = 1)
	- however on the two rectangles (sample index = 3) it doesn't work
	- should understand why and how it works on one case but not on the other case <- it was the learning rate
	- now we try updating the sigma too
		- it worked! the final config is `python losses.py vangool --n_iter=1000 --out_channels=4 --clean --loss_kwargs="{'print_iou' : False, 'per_instance':False, 'w_var':0.01, 'w_seed':0.01, 'exp_s':False, 'w_inst':1}" --sample_index=3 --lr=0.1`
	- now the options are: my custom loss, google loss. Google loss was sampling pixels and using a classifier. Let's do both. Kidding, google's is similar to the gaussian classifier. ours predicts (dx, dy, dr, dg, db, a,b,c,sigma,seed)
	- all of the losses were successfully used with a vitregs! these include hinge, offset, clusband, and ours. 
	- launch a training with ours (first overfit, then increase rectangles, then increase data) 

launched a moonshot run that might work
	[x] should add the git commit
	[x] should clean the taining script
	[x] should add finalization and cleaning to logs
	[x] should print the iou
	[x] should use less masks than rectangles 

- if we manage to have a nice training script that uses less masks than objects in the image then we should be able to start training with real images and expect a full segmentation
	- we can do clustering
	- we can take the segmentation for any given pixel
	- we can use the network along dino to have few-shot segmentation (clustering, etc.)
- check sigma vs object size
- check if hierarchy is automatically learned

My problem is now simpler: some rectangles of different colors, a network that outputs 10 channels, and a loss that makes 8 of those channels be a feature vector, and one of the other ones be a sigma. 2 of the feature channels are offset with the position of the pixels, and 3 are offset with the color. The other 3 feature channels are free. Then for some mask (one per visible rectangle) I compute the mean feature vector and the mean sigma and measure a (differentiable) IoU of the probs of a gaussian using those means thresholded at 0.5.., and maximize that.
I'm trying to make it work on the rectangles. The idea is that if I randomly compute the loss on some of the rectangles but not on all of them, then the network will still be forced to learn to segment them all, i.e. generate per-pixel features that give nice clusters. If this works then we could take whatever image segmentation dataset and train the network, and it should learn a relevant unsupervised segmentation. If this works then it's great. The next challenge is hierarchy.
