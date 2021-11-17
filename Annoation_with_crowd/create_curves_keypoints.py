import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xml.etree.ElementTree
from scripts.drawing.helper_plotting import *

MARKERS = ['-v', '-o', '-D', '-s', '-d', '-8', '-*', '-h', '-p', '-^', '-x', '-|'] * 2

QP = ['12', '17', '22', '27', '32', '37']
CODECS = ['VTM-10.0', 'HM-16.18']

PATH_DETERIORATED_SEQUENCES = '/home/fischer/DIV2K_compressed/'
DEFAULT_PATH = 'datasets/DIV2K_keypoints/evaluationResults/'
PATH_FOR_PLOTS = './plots/DIV2K_keypoints_crowd/'

MODEL_NAME = 'keypoint_rcnn_R_50_FPN_x3'
# MODEL_NAME = 'faster_rcnn_R_50_FPN_x3'

# define what script is supposed to create and calculate
PLOT_RESULTS = True
CALC_BDR = True

## Reload data from codec output files? Otherwise, the stored results are taken
# RELOAD = True
RELOAD = False

## setting for plotting data
OutputFileFormat = '.pdf'
# XSCALE = 'linear'
XSCALE = 'log'
FONTSIZE = 4

## settings for BDR calculation ######### watch out !! These have to fit to QP!! #############
# WHICH_BD = 'highQP'
WHICH_BD = 'smallQP'
# WHICH_BD = 'jvet'
# WHICH_BD = 'fullrange'
REF_CODEC = CODECS[0]		# Reference scaling list


def get_bitrate_and_vmaf(pathDetSequences, codec, reload=False, write=True):
	matrixBitrate = np.empty(len(QP), float)
	matrixVmaf = np.empty(len(QP), float)
	matrixPSNR = np.empty(len(QP), float)
	for qpIt, qp in enumerate(QP):
		if qp == 'uncompressed':
			matrixBitrate[qpIt] = np.Inf
			matrixVmaf[qpIt] = 100
			matrixPSNR[qpIt] = 99
		else:

			# import matrices from pickle file if available and return in order to save time
			prickleFileName = os.path.join(pathDetSequences, 'qp_' + qp,
										   'variables_' + codec + '_' + 'qp_' + qp + '.pkl')
			if os.path.isfile(prickleFileName) and not reload:
				with open(prickleFileName, 'rb') as f:
					# print('Load data from ' + prickleFileName)
					matrixBitrateQp, matrixVmafQp, matrixPSNRQp = pickle.load(f)
					matrixBitrate[qpIt] = matrixBitrateQp
					matrixVmaf[qpIt] = matrixVmafQp
					matrixPSNR[qpIt] = matrixPSNRQp
				continue

			startPath = os.path.join(pathDetSequences, 'qp_' + qp)
			walk = []
			for root, dirnames, filenames in os.walk(startPath):
				walk.append((root, filenames))

			# get needed files
			vtmFilenames = []
			vmafFilenames = []
			codecSmall = codec.partition('-')[0].lower()
			for root, filenames in walk:
				filenames.sort()
				# for filename in filenames[0:30]:
				for filename in filenames:
					if codecSmall + '_output.txt' in filename:
						vtmFilenames.append(os.path.join(root, filename))
					if 'vmaf_output.xml' in filename:
						vmafFilenames.append(os.path.join(root, filename))

			vtmFilenames.sort()
			vmafFilenames.sort()

			# get bitrate
			bitrateVector = np.zeros(len(vtmFilenames), float)
			# psnrVector = np.empty(len(vmafFilenames), float)
			for fIt, filename in enumerate(vtmFilenames):
				with open(filename, 'r') as file:
					rememberLine = 99999999
					for num, line in enumerate(file, 1):  # go through file line by line
						if 'Total Frames' in line:  # find line with PSNR value
							rememberLine = num + 1  # line with PSNR and bitrate comes two lines after this
						elif num == rememberLine:
							lineSplit = line.rsplit()
						# psnrVector[fIt] = lineSplit[3]  # for [6] YUV-PSNR, [3] for Y-PSNR
						# bitrateVector[fIt] = lineSplit[2]

						elif 'POC' in line:
							lineSplit = line.rsplit()
							poc = int(lineSplit[1])
							if 'HM' in codec:
								numBits = float(lineSplit[11])  # for HM
							else:
								if '9.0' in pathDetSequences or '10.0' in pathDetSequences:
									numBits = float(lineSplit[12])
								else:
									numBits = float(lineSplit[11])  # for VTM before 9.0
							bitrateVector[fIt] += numBits  # += so that the calculation is also valid for multiple frames

			matrixBitrate[qpIt] = np.mean(bitrateVector)
			# get the bitrate or num of bits in Mbit
			matrixBitrate[qpIt] = matrixBitrate[qpIt] / 1e6

			# get VMAF and PSNR
			vmafVector = np.empty(len(vmafFilenames), float)
			psnrVector = np.empty(len(vmafFilenames), float)
			for fIt, filename in enumerate(vmafFilenames):
				e = xml.etree.ElementTree.parse(filename).getroot()
				for atype in e.findall('fyi'):
					vmaf = atype.get('aggregateVMAF')
					vmafVector[fIt] = vmaf
					psnr = atype.get('aggregatePSNR')
					psnrVector[fIt] = psnr
			matrixVmaf[qpIt] = np.mean(vmafVector)
			matrixPSNR[qpIt] = np.mean(psnrVector)

		# save variables with pickle
		if write:
			with open(prickleFileName, 'wb') as f:
				pickle.dump([matrixBitrate[qpIt], matrixVmaf[qpIt], matrixPSNR[qpIt]], f)

	return matrixBitrate, matrixVmaf, matrixPSNR


def get_ap_values(codec):

	if codec == 'uncompressed':
		retDict = {}
		path = os.path.join(DEFAULT_PATH, 'uncompressed', MODEL_NAME)
		with open(os.path.join(path, 'resultKeypointDetection_uncompressed.txt'), 'r') as f:
			content =f.readlines()
			indxStartTable = []
			for lI, line in enumerate(content):
				if 'AP' in line:
					indxStartTable.append(lI)
			for indx in indxStartTable:
				curDict = {}
				type = content[indx-1].partition(':')[0].split(' ')[-1]
				apTypes = content[indx].split('|')
				apValues = content[indx+2].split('|')
				apTypesNew = []
				apValuesNew = []
				for apI, apType in enumerate(apTypes):
					if not 'AP' in apType:
						continue
					apTypesNew.append(apType.replace(' ', ''))
					apValuesNew.append(np.float(apValues[apI]))
					curDict[apType.replace(' ', '')] = np.float(apValues[apI])
				retDict[type] = curDict

		return retDict
	else:
		retDict = {}
		for qpI, qp in enumerate(QP):
			dictPerQP = {}
			path = os.path.join(DEFAULT_PATH, codec, MODEL_NAME)
			with open(os.path.join(path, 'resultKeypointDetection_qp_%s.txt' % qp), 'r') as f:
				content = f.readlines()
				indxStartTable = []
				for lI, line in enumerate(content):
					if 'AP' in line:
						indxStartTable.append(lI)
				for indx in indxStartTable:
					curDict = {}
					type = content[indx - 1].partition(':')[0].split(' ')[-1]
					apTypes = content[indx].split('|')
					apValues = content[indx + 2].split('|')
					apTypesNew = []
					apValuesNew = []
					for apI, apType in enumerate(apTypes):
						if not 'AP' in apType:
							continue
						apTypesNew.append(apType.replace(' ', ''))
						apValuesNew.append(np.float(apValues[apI]))
						curDict[apType.replace(' ', '')] = np.float(apValues[apI])
					dictPerQP[type] = curDict
				retDict[qp] = dictPerQP
		return retDict


def main():
	print()

	# matrixApsUncompressedBBox = np.empty((6), np.float)
	# matrixApsUncompressedKey = np.empty((5), np.float)
	matrixBitrate = np.empty((len(QP), len(CODECS)), np.float)
	matrixVmaf = np.empty((len(QP), len(CODECS)), np.float)
	matrixPsnr = np.empty((len(QP), len(CODECS)), np.float)
	# matrixApsBBox = np.empty((len(QP), len(CODECS), 6), np.float)
	# matrixApsKey = np.empty((len(QP), len(CODECS), 5), np.float)

	dictApUncompressed = get_ap_values('uncompressed')

	listDictApCompressed = []
	for cI, codec in enumerate(CODECS):
		pathToDetSeq = os.path.join(PATH_DETERIORATED_SEQUENCES, codec)
		matrixBitrate[:, cI], matrixVmaf[:, cI], matrixPsnr[:, cI] = get_bitrate_and_vmaf(pathToDetSeq, codec, RELOAD)
		dictApCompressed = get_ap_values(codec)
		listDictApCompressed.append(dictApCompressed)
		print()


	### plot results ###
	if PLOT_RESULTS:
		os.makedirs(PATH_FOR_PLOTS, exist_ok=True)

		set_rcparams_icip2018(1.3)
		# plt.figure()
		# for cI, codec in enumerate(CODECS):
		# 	plt.plot(matrixBitrate[:, cI], matrixVmaf[:, cI], MARKERS[cI], label=codec)
		# plt.xlabel('Avg Bitstream Size per Frame in MBit')
		# plt.ylabel('VMAF')
		# plt.xscale(XSCALE)
		# plt.legend(fontsize=FONTSIZE)
		# plt.savefig(PATH_FOR_PLOTS + 'VMAF_over_bitrate' + OutputFileFormat, bbox_inches='tight', pad_inches=0)
		#
		# plt.figure()
		# for cI, codec in enumerate(CODECS):
		# 	plt.plot(matrixBitrate[:, cI], matrixPsnr[:, cI], MARKERS[cI], label=codec)
		# plt.xlabel('Avg Bitstream Size per Frame in MBit')
		# plt.ylabel('PSNR in dB')
		# plt.xscale(XSCALE)
		# plt.legend(fontsize=FONTSIZE)
		# plt.savefig(PATH_FOR_PLOTS + 'PSNR_over_bitrate' + OutputFileFormat, bbox_inches='tight', pad_inches=0)

		usedMeasurements = list(dictApUncompressed.keys())
		for usedMeasurement in usedMeasurements:
			usedApTypes = list(dictApUncompressed[usedMeasurement].keys())
			for usedApType in usedApTypes:
				matrixAp = np.zeros((len(QP), len(CODECS)), np.float)
				for qpI, qp in enumerate(QP):
					for cI, codec in enumerate(CODECS):
						matrixAp[qpI, cI] = listDictApCompressed[cI][qp][usedMeasurement][usedApType]
				plt.figure()
				for cI, codec in enumerate(CODECS):
					plt.plot(matrixBitrate[:, cI], matrixAp[:, cI], MARKERS[cI], label=codec)
				plt.xlabel('Avg Bitstream Size per Frame in MBit')
				plt.ylabel(usedMeasurement + '' + usedApType)
				plt.xscale(XSCALE)
				plt.legend(fontsize=FONTSIZE)
				plt.savefig(os.path.join(PATH_FOR_PLOTS, usedMeasurement + '' + usedApType + '_over_bitrate' + OutputFileFormat), bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
