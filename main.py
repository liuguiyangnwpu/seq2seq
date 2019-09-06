from seq2seq.models import AttentionSeq2Seq
from src.dataset import NaskdaqDataGenerator, TimeSeriesNaskdaq
from keras.callbacks import ModelCheckpoint


input_length = 5
output_length = 3
n_dim = 3

samples = 100
hidden_dim = 24

nasdaq_time_series = TimeSeriesNaskdaq(input_length, output_length, n_dim)
nasdaq_time_series.load_data()
nasdaq_generator = NaskdaqDataGenerator(nasdaq_time_series, samples)


def main_train():
    # x = np.random.random((samples, input_length, input_dim))
    # y = np.random.random((samples, output_length, output_dim))
    checkpoint = ModelCheckpoint('model_best_weights.h5',
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=10)

    model = AttentionSeq2Seq(output_dim=n_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, n_dim), depth=4)
    model.compile(loss='mse', optimizer='sgd')

    model.fit_generator(generator=nasdaq_generator,
                        use_multiprocessing=True,
                        workers=5,
                        epochs=100000,
                        callbacks=[checkpoint])

main_train()
