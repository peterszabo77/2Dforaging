from parameters import *
from ForagingModel import *
from ReinforcementLearning import *
if WITHGUI:
	import wx
	from GUIPanels import *

if WITHGUI:
	class MyApp(wx.App):
		def __init__(self):
			wx.App.__init__(self, redirect=False)

		def OnInit(self):
			self.model=ForagingModel()
			self.rl=ReinforcementLearning(self.model)
			self.frame = MyFrame(None, rl=self.rl)
			return True
		
		def OnExitApp(self, evt):
			self.frame.Close(True)

	app = MyApp()
	app.MainLoop()
else:
	model=ForagingModel()
	rl=ReinforcementLearning(model)
	rl.StartLearning()
