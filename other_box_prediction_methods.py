def predict_boxes_near(heatmap,T,threshold=0.85):	#bounding box usually large in size
	threshold = threshold*np.amax(heatmap)
	t_area = T.shape[0]*T.shape[1]
	def explore(i,j,cnt):
		if heatmap[i][j]<threshold or cnt>100:
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
		
	boxes = []
	for i in range(len(heatmap)):
		for j in range(len(heatmap[0])):
			if heatmap[i][j]>threshold:
				coords = explore(i,j,0)
				tl_row,tl_col,br_row,br_col=min(coords[0]), min(coords[1]), max(coords[0]), max(coords[1])
				if 0.3*t_area <(tl_row-br_row)-(tl_col-br_col)<1.5*t_area and 0.6<=(tl_row-br_row)/(tl_col-br_col)<=1.6:
					score = np.mean(np.array(heatmap)[tl_row:br_row+1,br_row:br_col+1])
					boxes.append([tl_row,tl_col,br_row,br_col,score])
	return boxes
	
	
def predict_boxes(heatmap,T,threshold=0.91):      #easy to cause a series of overlapped false alarms on the edge of an object
	'''
	This function takes heatmap and returns the bounding boxes and associated
	confidence scores.
	'''

	'''
	BEGIN YOUR CODE
	'''
	threshold = threshold*np.amax(heatmap)
	temp_h = int(T.shape[0]//2)
	temp_w = int(T.shape[1]//2)
	boxes = []
	origin_map=np.copy(heatmap)
	
	center_r, center_c =-1, -1
	while True:
		max_val = np.amax(heatmap)
		if max_val<threshold:
			break
		center_posi = np.where(origin_map==max_val)    
		if center_r!= center_posi[0][0] and center_c!= center_posi[1][0]:
			center_r, center_c=center_posi[0][0],center_posi[1][0]
		else:
			break
		
		print(center_r,center_c)
				
		tl_row = max(center_r-temp_h,0)
		tl_col = max(center_c-temp_w,0)
		br_row = min(center_r+temp_h,len(heatmap))
		br_col = min(center_c+temp_w,len(heatmap[0]))
		
		for row in range(tl_row,br_row+1):
			for col in range(tl_col,br_col+1):
				heatmap[row][col] = 0
				
		score = origin_map[center_r][center_c]              ##### score: score of conv / temp conv temp
		boxes.append([tl_row,tl_col,br_row,br_col,score])
	 
	return boxes