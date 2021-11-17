import json



def main():
    fileName = '/home/fischer/git/detectron2/datasets/DIV2K_keypoints/labels/test_keypoint_with_crowd.json'
    with open(fileName, 'r') as f:
        data = json.load(f)

    data['categories'] = [data['categories'][0]]
    for anno in data['annotations']:
        if anno['category_id'] == 2:		# than it is 'person_crowd'
            anno['iscrowd'] = True
            anno['category_id'] = 1
            anno['keypoints'] = [0] * 51        # fill in dummy keypoint values
            anno['num_keypoints'] = 0

    with open('datasets/DIV2K_keypoints/labels/test_keypoint_with_crowd_converted.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
