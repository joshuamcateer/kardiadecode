Simple python script to decode audio recordings of Kardiamobile version 1 single channel ECG machine. 

The Kardiamobile ECG machine outputs an FM-modulated 19KHz audio signal that is modulated between 18KHz-20KHz with a calibration of 200Hz/mV. This code reads in audio signals and extracts the FM-modulated ECG signal.

This code does not a medical product. This is not tested in any way. This code should not be used to monitor or diagnose a medical condition.

I created this because the Karia API is not available to non-enterprise customers.

The Kardiamobile 6L is not compatible with this code as it uses Bluetooth not high-frequency sound waves.


To make recordings you can use the microphone of a smartphone. I used the Samsung voice recorder app. Set the voice recorder app to the highest quality available (highest sample rate and highest bit rate and two channels (or more) if available)).

The requirements can be pip installed. ReadECG.py is the script that can be used to generate the ecg signal.