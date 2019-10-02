import numpy as np
from parameters import *
import random
import itertools
from math import ceil, pi, sin, cos, atan2
import torch

def Dist2(x1,y1,x2,y2):
	xd=abs(x1-x2); xd=min(xd, L-xd)
	yd=abs(y1-y2); yd=min(yd, L-yd)
	return xd**2+yd**2

def SignedDirs(x1,y1,x2,y2):
	if abs(y2-y1)<L-abs(y2-y1):
		yd=y2-y1
	elif y2<y1:
		yd=y2-y1+L
	else:
		yd=y2-y1-L
	if abs(x2-x1)<L-abs(x2-x1):
		xd=x2-x1
	elif x2<x1:
		xd=x2-x1+L
	else:
		xd=x2-x1-L
	return (xd,yd)

class Forager():
	size=FORAGER_SIZE #size radius
	R=FORAGER_INTERACTION_RADIUS #interaction radius
	Rsq=R**2 #squared interaction radius
	speed=10.0
	interaction_filter=np.zeros((2, FORAGER_VISRESOL, FORAGER_VISRESOL)) # 1 within R, 0 othervise
	m=int(FORAGER_VISRESOL/2)
	cellsize=2*R/FORAGER_VISRESOL
	for i in range(FORAGER_VISRESOL):
		for j in range(FORAGER_VISRESOL):
			di=abs(i-m)
			dj=abs(j-m)
			if di!=0:
				di=(0.5+di-1)*cellsize
			if dj!=0:
				dj=(0.5+dj-1)*cellsize
			if di**2+dj**2<=Rsq:
				interaction_filter[0][i][j]=1
				interaction_filter[1][i][j]=1
			else:
				interaction_filter[0][i][j]=0
				interaction_filter[1][i][j]=0

	def __init__(self, size):
		self.x=random.random()*size
		self.y=random.random()*size
		self.direction=random.randrange(DIR_RESOL)*2*pi/DIR_RESOL #movement direction in radians
		self.newdirection=None
		self.dx=None
		self.dy=None
		self.gridcell=None
		self.reward=None
		self.oldcolliders=[]

	#required shape is (features, Lh, Lv)
	def GetVI(self, grid):
		#m=np.zeros((1, FORAGER_VISRESOL, FORAGER_VISRESOL))
		m=Forager.interaction_filter-1
		#m=m-1
		interactingFooditems=grid.GetNearbyFooditems(self)
		for fooditem in interactingFooditems:
			xd, yd=SignedDirs(self.x, self.y, fooditem.x, fooditem.y)
			i=int((Forager.R+xd) / (2*Forager.R/FORAGER_VISRESOL)) # distance from left / cellsize
			j=int((Forager.R+yd) / (2*Forager.R/FORAGER_VISRESOL)) # distance from bottom / cellsize
			m[0][i][j]=1
		interactingForagers=grid.GetNearbyForagers(self)
		for forager in interactingForagers:
			xd, yd=SignedDirs(self.x, self.y, forager.x, forager.y)
			i=int((Forager.R+xd) / (2*Forager.R/FORAGER_VISRESOL)) # distance from left / cellsize
			j=int((Forager.R+yd) / (2*Forager.R/FORAGER_VISRESOL)) # distance from bottom / cellsize
			m[1][i][j]=1
		return m

class FoodItem():
	birthRate=FOOD_BR
	decayRate=FOOD_DR
	size=FOOD_SIZE

	def __init__(self, sidelength):
		self.x=random.random()*sidelength
		self.y=random.random()*sidelength
		self.gridcell=None

class GridCell():
	def __init__(self, i, j):
		self.foragers=[]
		self.fooditems=[]
		self.i=i
		self.j=j
	
class Grid():
	def __init__(self):
		self.resolution=ceil(L/(2*FORAGER_INTERACTION_RADIUS))
		self.cellsize=L/self.resolution
		self.gridcells=[[GridCell(i,j) for j in range(self.resolution)] for i in range(self.resolution)]

	def AttachForagers(self, foragers):
		for forager in foragers:
			i = int(forager.x // self.cellsize)
			j = int(forager.y // self.cellsize)
			forager.gridcell = self.gridcells[i][j]
			self.gridcells[i][j].foragers.append(forager)

	def AttachFooditems(self, fooditems):
		for fooditem in fooditems:
			i = int(fooditem.x // self.cellsize)
			j = int(fooditem.y // self.cellsize)
			fooditem.gridcell = self.gridcells[i][j]
			self.gridcells[i][j].fooditems.append(fooditem)

	def UpdateForager(self, forager):
		i = int(forager.x // self.cellsize)
		j = int(forager.y // self.cellsize)
		if forager.gridcell != self.gridcells[i][j]:
			forager.gridcell.foragers.remove(forager)
			forager.gridcell = self.gridcells[i][j]
			forager.gridcell.foragers.append(forager)

	def RemoveFoodItem(self, fooditem):
		fooditem.gridcell.fooditems.remove(fooditem)

	def getNeighbourCells(self, gridcell):
		templist=[]
		for i, j in itertools.product((-1,0,1),(-1,0,1)):
			ni=(gridcell.i+i) % self.resolution
			nj=(gridcell.j+j) % self.resolution
			templist.append(self.gridcells[ni][nj])
		return templist

	def GetNearbyFooditems(self, forager):
		fooditems=[]
		distances=[]
		for ncell in self.getNeighbourCells(forager.gridcell):
			for fooditem in ncell.fooditems:
				distance2=Dist2(forager.x, forager.y, fooditem.x, fooditem.y)
				if distance2<Forager.Rsq:
					fooditems.append(fooditem)
					distances.append(distance2)
		fooditems=[fooditem for _, fooditem in sorted(zip(distances, fooditems))]
		return fooditems

	def GetNearbyForagers(self, forager):
		foragers=[]
		distances=[]
		for ncell in self.getNeighbourCells(forager.gridcell):
			for nforager in ncell.foragers:
				if nforager==forager:
					continue
				distance2=Dist2(forager.x, forager.y, nforager.x, nforager.y)
				if distance2<Forager.Rsq:
					foragers.append(nforager)
					distances.append(distance2)
		foragers=[nforager for _, nforager in sorted(zip(distances, foragers))]
		return foragers

class ForagingModel():
	def __init__(self):
		random.seed()
		self.Reset()
		
	def Reset(self):
		self.L=L # arena side length
		self.A=self.L**2 # arena area
		self.dT=DT

		# foragers
		self.foragers=[Forager(self.L) for i in range(N_FORAGERS)]

		# food items
		equilibrium_number=int(self.A*FoodItem.birthRate/FoodItem.decayRate)
		self.fooditems=[FoodItem(self.L) for i in range(equilibrium_number)]

		# grid
		self.grid=Grid()
		self.grid.AttachForagers(self.foragers)
		self.grid.AttachFooditems(self.fooditems)

	def Update(self, action_list):
		# forager speed vectors
		# collisions
		for forager in self.foragers:
			interactingForagers=self.grid.GetNearbyForagers(forager)
			collidedForagers=[fo for fo in interactingForagers if Dist2(forager.x, forager.y, fo.x, fo.y)<(2*Forager.size)**2]
			newcolliders=[fo for fo in collidedForagers if fo not in forager.oldcolliders]
			if ELASTIC_COLLISIONS and len(newcolliders)>0:
				otherforager=newcolliders[0]
				x_dist, y_dist = SignedDirs(forager.x, forager.y, otherforager.x, otherforager.y)
				theta=atan2(y_dist, x_dist)
				#alpha1=forager.direction
				#forager.direction=2*theta+pi-alpha1
				alpha1=forager.direction
				alpha2=otherforager.direction
				vx=cos(alpha2-theta)*cos(theta)+sin(alpha1-theta)*cos(theta+pi/2)
				vy=cos(alpha2-theta)*sin(theta)+sin(alpha1-theta)*sin(theta+pi/2)
				forager.newdirection=atan2(vy, vx)
			else:
				forager.newdirection=forager.direction
			forager.reward=-1*len(newcolliders)
			forager.oldcolliders=collidedForagers

		for i, forager in enumerate(self.foragers):
			# set action
			if len(forager.oldcolliders)==0 or not ELASTIC_COLLISIONS:
				forager.direction=action_list[i]*2*pi/DIR_RESOL
			else:
				forager.direction=forager.newdirection
			forager.dx=Forager.speed*cos(forager.direction)
			forager.dy=Forager.speed*sin(forager.direction)

		# consume fooditems
		for i, forager in enumerate(self.foragers):
			interactingFooditems=self.grid.GetNearbyFooditems(forager)
			consumedFooditems=[fi for fi in interactingFooditems if Dist2(forager.x, forager.y, fi.x, fi.y)<Forager.size**2]
			notconsumedfooditems=[fi for fi in interactingFooditems if fi not in consumedFooditems]
			forager.reward+=len(consumedFooditems)
			for fooditem in consumedFooditems:
				self.grid.RemoveFoodItem(fooditem)
				self.fooditems.remove(fooditem)

		# fooditems
		if FoodItem.birthRate*self.A*self.dT>1 or FoodItem.decayRate*self.dT>1:
			print('fooditem rates are too high')
		if random.random()<FoodItem.birthRate*self.A*self.dT:
			self.fooditems.append(FoodItem(self.L))
			self.grid.AttachFooditems([self.fooditems[-1]])
		decayedFooditems=[f for f in self.fooditems if random.random()<FoodItem.decayRate*self.dT]
		for fooditem in decayedFooditems:
			self.grid.RemoveFoodItem(fooditem)
			self.fooditems.remove(fooditem)

		# forager move
		for forager in self.foragers:
			forager.x=(forager.x+forager.dx*self.dT) % self.L
			forager.y=(forager.y+forager.dy*self.dT) % self.L
			self.grid.UpdateForager(forager)

	#returns the visual information as a two dimensional array with one feature column
	#required shape is (batchsize, features, Lh, Lv)
	def GetVIs(self):
		VIs=[]
		for forager in self.foragers:
			#VIs.append(torch.from_numpy(forager.GetVI(self.grid)).type('torch.FloatTensor'))
			VIs.append(forager.GetVI(self.grid))
		#return torch.stack(VIs, dim=0)
		return np.stack(VIs, axis=0)

	def GetRewards(self):
		return [forager.reward for forager in self.foragers]
