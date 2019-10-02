import sys
import wx
from wx import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import gluNewQuadric, gluDisk
from OpenGL.GLUT import *
from math import radians, degrees, sin, cos, pi
from ForagingModel import *
from ReinforcementLearning import *
from parameters import *

class MyFrame(wx.Frame):
	def __init__(self, parent, rl):
		super(MyFrame, self).__init__(parent, title="2D Foraging Model", style=wx.DEFAULT_FRAME_STYLE, size=(400, 400))
		self.rl=rl
		self.model=rl.model
		self.panel=MainPanel(self, self.rl, self.model)
		self.rl.SetCanvas(self.panel.modelCanvas)
		self.Show(True)
		wx.CallLater(1000, self.rl.StartLearning)
		
class MyCanvasBase(glcanvas.GLCanvas):
	def __init__(self, parent):
		glcanvas.GLCanvas.__init__(self, parent, -1)
		self.init = False
		self.context = glcanvas.GLContext(self)
		
		self.size = None
		self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
		self.Bind(wx.EVT_SIZE, self.OnSize)
		self.Bind(wx.EVT_PAINT, self.OnPaint)

	def OnEraseBackground(self, event):
		pass # Do nothing, to avoid flashing on MSW.

	def OnSize(self, event):
		wx.CallAfter(self.DoSetViewport)
		event.Skip()

	def DoSetViewport(self):
		size = self.size = self.GetClientSize()
		self.SetCurrent(self.context)
		glViewport(0, 0, size.width, size.height)
		
	def OnPaint(self, event):
		dc = wx.PaintDC(self)
		self.SetCurrent(self.context)
		if not self.init:
			self.InitGL()
			self.init = True
		self.OnDraw()

class ModelCanvas(MyCanvasBase):
	def __init__(self, parent, model):
		MyCanvasBase.__init__(self, parent)
		self.model=model
		self.drawing=DRAWING
		self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)

	def InitGL(self):
		# set viewing projection
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, self.model.L, 0, self.model.L, -1, 1)
		self.quadricObject=gluNewQuadric()

	def OnLeftUp(self, e):
		if self.drawing:
			self.drawing=False
		else:
			self.drawing=True

	def OnDraw(self):
		if not self.drawing:
			return
		# clear color and depth buffers
		glClear(GL_COLOR_BUFFER_BIT)

		posx=50.0
		posy=50.0
		direction=45
		size=10
		
		#draw food
		glColor3f(0.0, 0.0, 1.0);
		for food in self.model.fooditems:
			# position viewer
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()

			# position object
			glTranslatef(food.x, food.y, 0) #translation third
			glScalef(food.size, food.size, food.size) #scale second
		
			gluDisk(self.quadricObject, 0, 0.5, 16,1)

		#draw animals
		glColor3f(1.0, 1.0, 1.0);
		for forager in self.model.foragers:
			# position viewer
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()

			# position object
			glTranslatef(forager.x, forager.y, 0) #translation third
			glScalef(forager.size, forager.size, forager.size) #scale second
			#glRotatef(degrees(particle.rhoRad), 0, 0, 1) #rotation first
		
			gluDisk(self.quadricObject, 0, 0.5, 16,1)
			
		self.SwapBuffers()

class MainPanel(wx.Panel):
	def __init__(self, parent, rl, model):
		wx.Panel.__init__(self, parent, -1)
		self.rl=rl
		self.model=model
		
		self.modelCanvas = ModelCanvas(self, model)
		self.modelCanvas.SetMinSize((200, 200))

		boxsizer=wx.BoxSizer(wx.HORIZONTAL)
		boxsizer.Add(self.modelCanvas, 3, wx.SHAPED | wx.ALIGN_CENTER, 0)
		
		self.SetSizer(boxsizer)
		self.SetAutoLayout(True)

