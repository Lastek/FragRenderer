'''
Created on Nov 11, 2013

@author: Owner
'''
import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
#from compiler.ast import For
sys.path += ['.']

def readPalette(filename):
    data = []
    f = open( filename, 'r' ) #16 color palette
    f.seek(0,2)
    j = f.tell()
    f.seek(0)
    
    i = 0
    f.read(12) #skip stuff we dont care about
    j -= 12 
#    data.append( f.read( 3 ) )
#    f.read(4)
    
    for i in xrange( j/3 ):
        for x in xrange(3):
            data.append( f.read( 1 ) )
    f.close()
    return data
    
#def resize():
frac = None
MAX_ITER = 20
palette16 = []
rend = None

class Fractal():
    def __init__(self, size):
        self.array = np.zeros(shape=(size[0], size[1]))
        self.width = size[0]
        self.height = size[1]
        self.color = np.zeros(size[0]*size[1])
        #ideally i should normalize the min/max based on screen aspect
        self.r_min, self.r_max = -2.0, 2.0
        self.i_min, self.i_max = -2.0, 2.0
        self.c = complex(1.0, 0.440)
        self.r_range = np.arange(self.r_min, self.r_max, 4./self.width)
        self.i_range = np.arange(self.i_max, self.i_min, 4./self.height)
         
    def build(self):
        index = 0
        for i in self.i_range:
            for r in self.r_range:
                iter = 0
                z = complex(r, i) 
                while abs(z.real) < 50 and iter < MAX_ITER:
                    z = z*z+self.c
                    iter += 1
                self.color[index] = iter*10
                index+=1
        return self.color
        
    def rebuild(self):
        pass
        
glsl_v = \
"""
#version 420
//uniform vec3 pos;
//mat4 idm = mat4(1.0);
layout(location = 0) in vec2 VertPos;
varying vec2 pos;
uniform int step;
uniform vec2 pan;
uniform float zoom;
void main() {
    //pos = vec2((VertPos.x-pan.x), (VertPos.y-pan.y))/zoom;
    pos = vec2((VertPos.x*(1.0/zoom))+pan.x, (VertPos.y* (1.0/zoom))+pan.y);
    gl_Position = vec4(VertPos.x, VertPos.y, 0.0f, 1.0f);
    
}
"""
glsl_f = \
"""
#version 420
uniform sampler1D tex;
uniform int step;
uniform vec2 frac;
varying vec2 pos;
uniform bool palette;

vec2 c_mul(vec2 v1, vec2 v2) {
    return vec2((v1.x*v2.x)-(v1.y*v2.y), (v1.x*v2.y)+(v1.y*v2.x));
}

void main() {
    //float norm = sqrt(pos.x*pos.x+pos.y*pos.y);
    //vec3 color = vec3(pos.x*pos.x/norm, pos.y*pos.y/norm, pos.x*pos.y/norm);
    //gl_FragColor = vec4(color, 1.0);
    vec2 c = /*vec2(0.8, 0.08) + */frac;
    int iter = 127 + step;
    vec2 z = pos;
    //z.x = 2.0 * (gl_TexCoord[0].x - 0.5);//pos;
    //z.y = 2.0 * (gl_TexCoord[0].y - 0.5);
    float i = 0;
    for(i = 0; i < iter; ++i) {
        z = c_mul(z, z);
        z = vec2(z.x + c.x, z.y + c.y);
        if(dot(z,z) > 4.0) break;
    }
    if(palette == false)
        gl_FragColor = vec4(i*.04, i*0.008, i*0.001, 0.0);
    else
        gl_FragColor = texture(tex, (i == iter ? 32.0/i : i/16.0));
}
"""

class renderer():
    def __init__(self, size, palette):
        
        self.drawing_area = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype='float32')
        self.shaderProg = GLuint
        self.vbo = GLuint
        self.ebo = GLuint
        self.tex = GLuint
        self.size = [size[0], size[1]]
        self.palette_colors = len(palette)/3
        self.palette = np.empty(len(palette), dtype='float32')
        print len(palette)
        #normalize
        for i in xrange(len(palette)):
            self.palette[i] = float(ord(palette[i])/255.0)
            print ord(palette[i])
        for i in xrange(48):
            print self.palette[i]
        
        self.palette_bool = False
        self.key_zoom = 1.0
        self.key_panx = 0.0
        self.key_pany = 0.0
        self.key_pan_precis = 0.01
        self.key_step = 1
        self.uni_c = GLuint
        self.c_1 = 0.80
        self.c_2 = 0.08
        
        self.count = 0
        
        self.hud1_str = ["zoom: %f", "pos x: %f  y: %f", "iterations: %i",\
                         "c(%f, %fi)", "palette size: %i"]
        self.hud2_str = ["[: show / hide stats", "]: show / hide help", "Q: zoom in", "E: zoom out", "WSAD: pan",\
                         "-+: pan precision", "ZX: -+ iterations", "FG: -+ c-plane x coord", "VB: -+ c-plane i coord",\
                         "H: custom color palette on/off" ]
        self.hud1_size = len(self.hud1_str)
        self.hud2_size = len(self.hud2_str)
        self.hud1_pos = (-0.98, 0.97)
        self.hud2_pos = (-0.98, -0.57)
        self.hud1_show = 1
        self.hud2_show = 1
        self.hud_ds = [""]*(self.hud2_size+self.hud2_size)
        self.initialize()
        
    def keyboard(self, key, x, y):
        if(key == '['):
            self.hud1_show ^= 1
        elif(key == ']'):
            self.hud2_show ^= 1
        elif(key == '-'):
            self.key_pan_precis *= 10
        elif(key == '+'):
            self.key_pan_precis /= 10
        elif(key == 'e'):
            self.key_zoom += 0.1
            self.key_zoom *= 1.2
        elif(key == 'q'):
            self.key_zoom -= 0.1
            self.key_zoom *= 0.83
        elif(key == 'w'):
            self.key_panx += self.key_pan_precis/self.key_zoom*1.5
        elif(key == 's'):
            self.key_panx -= self.key_pan_precis/self.key_zoom*1.5
        elif(key == 'a'):
            self.key_pany -= self.key_pan_precis/self.key_zoom*1.5
        elif(key == 'd'):
            self.key_pany += self.key_pan_precis/self.key_zoom*1.5
        elif(key == 'z'):
            self.key_step -= 1
        elif(key == 'x'):
            self.key_step += 1
        elif(key == 'f'):
            self.c_1 -= 0.001
        elif(key == 'g'):
            self.c_1 += 0.001
        elif(key == 'v'):
            self.c_2 -= 0.001
        elif(key == 'b'):
            self.c_2 += 0.001
        elif(key == 'h'):
            if(self.palette_bool == False):
                self.palette_bool = True
            else: 
                self.palette_bool = False
        
        glutPostRedisplay()
        #print "zoom %f" % self.key_zoom
                         
    def createBuffers(self):
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, len(self.drawing_area)*8, self.drawing_area, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        
    def createShaders(self):
        status = 0
        log = ""
        
        VS = glCreateShader(GL_VERTEX_SHADER)
        FS = glCreateShader(GL_FRAGMENT_SHADER)
        
        glShaderSource(VS, glsl_v)
        glShaderSource(FS, glsl_f)
        
        #compile
        glCompileShader(VS)
        status = glGetShaderiv(VS, GL_COMPILE_STATUS)
        if(status != GL_TRUE):
            print "Failed to compile Vertex Shader"
            log = glGetShaderInfoLog(VS)
            print "LOG:\n%s" % log 
            return -1
        else: print "Vertex Shader compiled successfully."
        
        glCompileShader(FS)
        status = glGetShaderiv(FS, GL_COMPILE_STATUS) 
        if(status != GL_TRUE):
            print "Failed to compile Fragment Shader"
            log = glGetShaderInfoLog(FS)
            print "LOG:\n%s" % log
            return -1
        else: print "Fragment Shader compiled successfully."
        
        self.shaderProg = glCreateProgram()
        
        glAttachShader(self.shaderProg, VS)
        glAttachShader(self.shaderProg, FS)
        
        glLinkProgram(self.shaderProg)
        
        status = glGetProgramiv(self.shaderProg, GL_LINK_STATUS)
        if(status != GL_TRUE):
            print "Failed to link shader program"
            log = glGetProgramInfoLog(self.shaderProg)
            print "LOG:\n%s" % log
            return -1
        else: print "Program linked successfully."
        #cleanup
        glDetachShader(self.shaderProg, VS)
        glDetachShader(self.shaderProg, FS)
        glDeleteShader(VS)
        glDeleteShader(FS)
        #
        glUseProgram(self.shaderProg)
        return 0
        
    def createTextures(self):
        self.tex = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_1D, self.tex)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, self.palette_colors, 0, GL_RGB, GL_FLOAT, self.palette)
        glUniform1i(glGetUniformLocation(self.shaderProg, 'tex'), 0)
        
    def initUniforms(self):        
        self.uni_pos = glGetUniformLocation(self.shaderProg, "pos")
        self.uni_step = glGetUniformLocation(self.shaderProg, "step")
        glUniform1i(self.uni_step, 0)
        self.uni_zoom = glGetUniformLocation(self.shaderProg, "zoom")
        glUniform1f(self.uni_zoom, 1.0)
        self.uni_pan = glGetUniformLocation(self.shaderProg, "pan")
        glUniform2f(self.uni_pan, 0, 0)
        self.uni_c = glGetUniformLocation(self.shaderProg, "frac")
        glUniform2f(self.uni_c, self.c_1, self.c_2)
        self.uni_palette = glGetUniformLocation(self.shaderProg, "palette")
        glUniform1i(self.uni_palette, self.palette_bool)
        
    def colorCycle(self):
        self.count += 3
        if(self.count > 47):
            self.count = 0
            
    def drawHud(self):
        if(self.hud1_show == True):
            self.hud_ds[0] = self.hud1_str[0] % (self.key_zoom)
            self.hud_ds[1] = self.hud1_str[1] % (self.key_panx, self.key_pany)
            self.hud_ds[2] = self.hud1_str[2] % (self.key_step+127)
            self.hud_ds[3] = self.hud1_str[3] % (self.c_1, self.c_2)
            if(self.palette_bool == True):
                self.hud_ds[4] = self.hud1_str[4] % (self.palette_colors)
            else:
                self.hud_ds[4] = ""
            glRasterPos2f(self.hud1_pos[0], self.hud1_pos[1])       
            
            for i in xrange(self.hud1_size):
                glRasterPos2f(self.hud1_pos[0], self.hud1_pos[1]-0.03*float(i))
                for x in range(len(self.hud_ds[i])):
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(self.hud_ds[i][x]))
        
        if(self.hud2_show == True):
            for j in xrange(self.hud2_size):
                self.hud_ds[self.hud1_size+j] = self.hud2_str[j]
            glRasterPos2f(self.hud2_pos[0], self.hud2_pos[1])
            
            for i in range(self.hud1_size, self.hud2_size+self.hud1_size):
                glRasterPos2f(self.hud2_pos[0], self.hud2_pos[1]-0.03*float(i))
                for x in range(len(self.hud_ds[i])):
                    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(self.hud_ds[i][x]))
                    
    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shaderProg)
        glUniform3f(self.uni_pos,
                    self.palette[self.count],
                    self.palette[self.count+1],
                    self.palette[self.count+2])
        glUniform1i(self.uni_step, self.key_step)
        glUniform1f(self.uni_zoom, self.key_zoom)
        glUniform2f(self.uni_pan, self.key_pany, self.key_panx) 
        glUniform2f(self.uni_c, self.c_1, self.c_2)
        glUniform1i(self.uni_palette, self.palette_bool)       
        glDrawArrays(GL_QUADS, 0, 4)
#        self.colorCycle()
        glUseProgram(0)
        self.drawHud()
                
        glutSwapBuffers()
        
        
    def initialize(self):
        glViewport(0, 0, self.size[0], self.size[1])
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.createBuffers()
        if(self.createShaders() == -1):
            print "**EXIT**"
            sys.exit()
        self.createTextures()
        self.initUniforms()
        self.draw()
            
def init():
    global frac
    global palette16
    global rend
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 800)
    glutCreateWindow("FragRenderer")
    #frac = Fractal((GLUT_WINDOW_WIDTH, GLUT_WINDOW_HEIGHT))
    palette16 = readPalette(sys.argv[1])
    print palette16
    rend = renderer((800, 800), palette16)
    
#    glutReshapeFunc(resize)
    glutDisplayFunc(rend.draw)
    glutKeyboardFunc(rend.keyboard)
    glutMainLoop()

init()