def predict_boxes_connect(heatmap,T,threshold=0.9):
	'''
	This function takes heatmap and returns the bounding boxes and associated
	confidence scores.
	'''

	'''
	BEGIN YOUR CODE
	'''
	
	temp_h=int(T.shape[0]//2)
	temp_w=int(T.shape[1]//2)
	print(temp_h)
	origin_map = np.copy(heatmap)
	
	def explore(i,j,cnt):
		if heatmap[i][j]<threshold or cnt>10000:
			return [[],[]]
		
		heatmap[i][j]=0
		coords = [[i],[j]]
		if i>=1:
			res1 = explore(i-1,j,cnt+1)
			coords[0]+=res1[0]
			coords[1]+=res1[1]
		if i<len(heatmap)-1:
			res2 = explore(i+1,j,cnt+1)
			coords[0]+=res2[0]
			coords[1]+=res2[1]
		if j>=1:
			res3 = explore(i,j-1,cnt+1)
			coords[0]+=res3[0]
			coords[1]+=res3[1]
		if j<len(heatmap[0])-1:
			res4 = explore(i,j+1,cnt+1)
			coords[0]+=res4[0]
			coords[1]+=res4[1]
		return coords
		
	connected_area = []
	for i in range(len(heatmap)):
		for j in range(len(heatmap[0])):
			if heatmap[i][j]>=threshold:
				coords = explore(i,j,0)
				tl_row = min(coords[0])
				tl_col = min(coords[1])
				br_row = max(coords[0])
				br_col = max(coords[1])
				connected_area.append([tl_row,tl_col,br_row,br_col]) 
	#print(connected_area)
				
	boxes_set = set()
	for tl_row,tl_col,br_row,br_col in connected_area:               
		max_conv = np.amax(origin_map[tl_row:br_row+1, tl_col:br_col+1])
		#print(origin_map[tl_row:br_row+1, tl_col:br_col+1])
		#print(max_conv)
		center_posi = np.where(origin_map==max_conv)
		#print(center_posi)
		
		center_r, center_c = -1, -1
		for r,c in zip(center_posi[0], center_posi[1]):
			if tl_row<=r<=br_row and tl_col<=r<=br_col:
				center_r, center_c=r, c 
		print(center_r,center_c)
		if center_r<0:
			continue
				
		tl_row = max(center_r-temp_h,0)
		tl_col = max(center_c-temp_w,0)
		br_row = min(center_r+temp_h,len(heatmap))
		br_col = min(center_c+temp_w,len(heatmap[0]))
		score = origin_map[center_r][center_c]      ##### score: score of conv / temp conv temp
		boxes_set.add((tl_row,tl_col,br_row,br_col,score))
	 
	boxes = [list(x) for x in boxes_set]

	return boxes