import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

MODEL_PATH = 'C:\\Users\\salag\\OneDrive\\Desktop\\final\\projectb10\\recognition\\mf.keras'
model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(learning_rate=1.000000082740371e-07),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

CLASSES = ['BabyCrawling', 'Biking', 'HorseRiding', 'Nunchucks', 'PlayingGuitar', 'Punch', 'SkateBoarding', 'WalkingWithDog', 'WritingOnBoard', 'carcrash', 'fighting']