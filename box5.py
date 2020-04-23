def boxNear(matrix, threshold):	
	def explore(i,j):
		if matrix[i][j]<threshold:
			return [[],[]]
		
		matrix[i][j]=threshold-0.1
		coords = [[i],[j]]
		if i>=1:
			res1 = explore(i-1,j)
			coords[0]+=res1[0]
			coords[1]+=res1[1]
		if i<matrix.shape[0]-1:
			res2 = explore(i+1,j)
			coords[0]+=res2[0]
			coords[1]+=res2[1]
		if j>=1:
			res3 = explore(i,j-1)
			coords[0]+=res3[0]
			coords[1]+=res3[1]
		if j<matrix.shape[1]-1:
			res4 = explore(i,j+1)
			coords[0]+=res4[0]
			coords[1]+=res4[1]
		return coords
		
	boxes = []
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if matrix[i][j]==1:
				coords = explore(i,j)
				boxes.append([min(coords[0]), min(coords[1]), max(coords[0]), max(coords[1])])
	return boxes