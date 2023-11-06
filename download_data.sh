mkdir sample_data
mkdir output output/release output/release/pacer output/release/pacer_group_cnn output/release/pacer_getup output/release/pacer_no_shape
gdown https://drive.google.com/uc?id=12xDUruRQHMMeuYhFh9ON3AVzXPyMOShT -O  sample_data/ # filtered shapes from AMASS
gdown https://drive.google.com/uc?id=1rmxR4I2_bHazkaTujV48R1xZSVaK5tjb -O  sample_data/ # sample standing neutral data.
gdown https://drive.google.com/uc?id=1rUSjVUYE8bIPZOyhbdQrKZ9-otE3fHpr -O  sample_data/ # amass_occlusion_v2

gdown https://drive.google.com/uc?id=1CF8mA9L3cAAOdDk6qtwDnAOTrQI8mien -O  output/release/pacer/ # pacer
gdown https://drive.google.com/uc?id=1xoHaO2Ig0S912lZbKgqRZSIixB8oUFeC -O  output/release/pacer_group_cnn/ # pacer_group_cnn
gdown https://drive.google.com/uc?id=1aPpHPypkqdbYCq8qVfzgpezyowvi5MR6 -O  output/release/pacer_getup/ # pacer_getup
gdown https://drive.google.com/uc?id=1srmXSH8c-4Recy1g4jgWCNzZhDQ7mWfH -O  output/release/pacer_no_shape/ # pacer_no_shape

