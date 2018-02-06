from nn import nn

import cpp_piano_learning_cnn_data_provider as provider
dataProvider = provider.PianoLearnerDataProvider()

CHECKPOINT_DIR = '/var/tmp/pls/checkpoints/'

lowest_loss = 10

nnSess = nn.NNSession()

for i in range(500000):
	
	batch_xs, batch_ys = dataProvider.getTrainingBatch(30)

	training_loss = nnSess.train(batch_xs, batch_ys, 0.5)

	if i % 10 == 0:

		test_xs, test_ys = dataProvider.getMiniTestData()
		test_loss = nnSess.getLoss(test_xs, test_ys, 1.0)

		print(i, ':', 'loss from training', training_loss, ': loss from test', test_loss)
		
		if test_loss < lowest_loss:
			lowest_loss = test_loss
			print('--------------------------> new record at num:', lowest_loss)
			filename = CHECKPOINT_DIR + str(i) + '-' + str(test_loss) + 'piano-learning-stream.ckpt'
			nnSess.saveCheckpoint(filename)
