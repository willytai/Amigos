import sys

if len(sys.argv) != 2:
	print ('Usage: python3 header.py <type>')
	print ('type:  EEG, ECG, GSR')
	sys.exit()


if sys.argv[1] == 'EEG':
	header = ',AF3_theta,AF3_slow alpha,AF3_alpha,AF3_beta,AF3_gamma,AF4_theta,AF4_slow alpha,AF4_alpha,AF4_beta,AF4_gamma,asymmetry_theta,asymmetry_slow alpha,asymmetry_alpha,asymmetry_beta,asymmetry_gamma,F7_theta,F7_slow alpha,F7_alpha,F7_beta,F7_gamma,F8_theta,F8_slow alpha,F8_alpha,F8_beta,F8_gamma,asymmetry_theta,asymmetry_slow alpha,asymmetry_alpha,asymmetry_beta,asymmetry_gamma,F3_theta,F3_slow alpha,F3_alpha,F3_beta,F3_gamma,F4_theta,F4_slow alpha,F4_alpha,F4_beta,F4_gamma,asymmetry_theta,asymmetry_slow alpha,asymmetry_alpha,asymmetry_beta,asymmetry_gamma,FC5_theta,FC5_slow alpha,FC5_alpha,FC5_beta,FC5_gamma,FC6_theta,FC6_slow alpha,FC6_alpha,FC6_beta,FC6_gamma,asymmetry_theta,asymmetry_slow alpha,asymmetry_alpha,asymmetry_beta,asymmetry_gamma,T7_theta,T7_slow alpha,T7_alpha,T7_beta,T7_gamma,T8_theta,T8_slow alpha,T8_alpha,T8_beta,T8_gamma,asymmetry_theta,asymmetry_slow alpha,asymmetry_alpha,asymmetry_beta,asymmetry_gamma,P7_theta,P7_slow alpha,P7_alpha,P7_beta,P7_gamma,P8_theta,P8_slow alpha,P8_alpha,P8_beta,P8_gamma,asymmetry_theta,asymmetry_slow alpha,asymmetry_alpha,asymmetry_beta,asymmetry_gamma,O1_theta,O1_slow alpha,O1_alpha,O1_beta,O1_gamma,O2_theta,O2_slow alpha,O2_alpha,O2_beta,O2_gamma,asymmetry_theta,asymmetry_slow alpha,asymmetry_alpha,asymmetry_beta,asymmetry_gamma,arousal,valence'

	file = open('EEG.csv', 'a+')
	file.write(header+'\n')
	file.close()

elif sys.argv[1] == 'ECG':
	header = ',rms_IBI,mean_IBI,std_IBI,skew_IBI,kur_IBI,mean+-std_IBI,mean_HR,std_HR,HRV,skew_HR,kur_HR,mean+-std_HR,low,mid,high,LF/HF'
	for i in range(60):
		header += ',sp 0-6'
	header += ',arousal,valence'
	file = open('ECG.csv', 'a+')
	file.write(header+'\n')
	file.close()

elif sys.agrv[1] == 'GSR':
	pass