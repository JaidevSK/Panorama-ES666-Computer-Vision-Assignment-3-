import I1_Stitcher
import I2_Stitcher
import I3_Stitcher
import I4_Stitcher
import I5_Stitcher
import I6_Stitcher

class PanaromaStitcher():
    def __init__(self):
        pass
    
    def stitch_Ix(Ix):
        if Ix == 'I1':
            I1_Stitcher.start()
        elif Ix == 'I2':
            I2_Stitcher.start()
        elif Ix == 'I3':
            I3_Stitcher.start()
        elif Ix == 'I4':
            I4_Stitcher.start()
        elif Ix == 'I5':
            I5_Stitcher.start()
        elif Ix == 'I6':
            I6_Stitcher.start()
        else:
            print('Invalid Input')
            return

    def make_all_panorama():
        all_images = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
        for Ix in all_images:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Stitching Panaroma for Image: ", Ix)
            PanaromaStitcher.stitch_Ix(Ix)

if __name__ == '__main__':
    PanaromaStitcher.make_all_panorama()
    print('All Panaroma Images are created')
