function imgN = readImg (filename),

	img = imread(filename);
	imgV = double (img (:)');
	imgN = featureNormalize(imgV');
	imgN = imgN';
end;
