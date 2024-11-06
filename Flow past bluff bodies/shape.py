import numpy as np
def shape(gridy,gridx,locX,locY,charL,body='circle'):
    booleanpoints = np.full((gridy, gridx), False)
    if body == 'circle':
        for x in np.arange(0,gridx,1):
            for y in np.arange(0,gridy,1):
                if ((x - locX)**2 + (y - locY)**2 <= (charL/2)**2):
                    booleanpoints[y][x] = True
    elif body == 'square':
        for x in np.arange(0,gridx,1):
            for y in np.arange(0,gridy,1):
                if ((x - locX<=charL/2 and x - locX >= (-1)*charL/2) and ((y - locY<=charL/2 and y - locY >= (-1)*charL/2))):
                    booleanpoints[y][x] = True

    elif body == 'prismdeg0':
        for x in np.arange(0,gridx,1):
            for y in np.arange(0,gridy,1):
                if (((1/np.sqrt(3)*(x-locX))-(y-locY) >= 0) and ((1/np.sqrt(3)*(x-locX))+(y-locY) >= 0) and (x-locX <= (np.sqrt(3)/2)*charL)):
                    booleanpoints[y][x] = True

    return booleanpoints