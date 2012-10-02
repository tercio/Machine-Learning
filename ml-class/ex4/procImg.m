function NORM = procImg (imgMatrixFile)

	load (imgMatrixFile);
    imgD = double(img);
	NORM = featureNormalize (imgD(:));
end;
