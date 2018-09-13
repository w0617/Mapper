#include "include/reconstructionthread.h"

ReconstructionThread::ReconstructionThread(CalibrationParameters *calib, StereoImage *stereo, QObject *parent)
    : QThread(parent)
    ,m_calib(calib)
    ,m_stereo(stereo)
{
    m_stereoImage = 0;

    //it is the max map point distance
    //if any point is far from the camera over this number, it would not be shown
    //TODO: make it could be changed in the mainwindow
    m_showingMinDistance = 0.1;
    m_showingMaxDistance = 20;

    if (m_calib->CheckRunDataType() == m_calib->DADAO_DATASET_FLAG)
    {
        m_f = m_calib->m_dadaoCalibParam.m_f; // x focal length in pixels.
        m_cu = m_calib->m_dadaoCalibParam.m_cu; // principal point (u-coordinate) in pixels
        m_cv = m_calib->m_dadaoCalibParam.m_cv; // principal point (v-coordinate) in pixels
        m_baseline = m_calib->m_dadaoCalibParam.m_baseline; // baseline in meters
    }
    else if (m_calib->CheckRunDataType() == m_calib->KITTI_DATASET_FLAG)
    {
        m_f = m_calib->m_KittiCalibParam.m_f; // x focal length in pixels.
        m_cu = m_calib->m_KittiCalibParam.m_cu; // principal point (u-coordinate) in pixels
        m_cv = m_calib->m_KittiCalibParam.m_cv; // principal point (v-coordinate) in pixels
        m_baseline = m_calib->m_KittiCalibParam.m_baseline; // baseline in meters
    }
    else if (m_calib->CheckRunDataType() == m_calib->LOITOR_DATASET_FLAG)
    {
        m_f = m_calib->m_loitorCalibParam.m_f; // x focal length in pixels.
        m_cu = m_calib->m_loitorCalibParam.m_cu; // principal point (u-coordinate) in pixels
        m_cv = m_calib->m_loitorCalibParam.m_cv; // principal point (v-coordinate) in pixels
        m_baseline = m_calib->m_loitorCalibParam.m_baseline; // baseline in meters
    }
    
    
    // init the compute disparity mode parameters.
    this->sps.initSpsParameter();
    this->bm.initBmParameter();
    this->sgbm.initSgbmParameter();
    // elas
    this->param.postprocess_only_left = true;
    this->param.filter_adaptive_mean = true;
    this->param.support_texture = 30;

    //MonoDepth
    //=======================Python MonoDepth======================
    Py_Initialize();
    
    string projectPath, mainModulePath, kittiModulePath;
    char *path;
    projectPath     = getcwd(path, 100);
    mainModulePath  = "sys.path.append('" + projectPath + "/Disparity/MonoDepth/')";
    kittiModulePath = projectPath + "/Disparity/MonoDepth/model_kitti/model_kitti" ;
    // Find monodepth.py
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(mainModulePath.data());  
            
    // import monodepth.py
    this->pModule = PyImport_ImportModule("monodepth");
    if (this->pModule != NULL) 
    {   // find def main() 
        this->pFunc   = PyObject_GetAttrString(this->pModule, "main");
        this->pArg = PyTuple_New(2);
    }
    else
        printf("\n\nCan't import monodepth !\n\n ");
    
    // setting the --checkpoint_path 
    this->pName = PyString_FromString(kittiModulePath.data());
    PyTuple_SetItem(this->pArg, 1, this->pName);
}

ReconstructionThread::~ReconstructionThread()
{
    if (m_stereoImage!=0)
    {
        delete m_stereoImage;
        m_stereoImage = 0;
    }
    m_mapPoints.clear();
}

void ReconstructionThread::SendImageAndPose(StereoImage::StereoImageParameters &s, libviso2_Matrix H)
{
    StereoImage::cameraFocus = m_f;
    StereoImage::cameraBaseline = m_baseline;

    if (m_stereoImage!=0)
    {
        delete m_stereoImage;
        m_stereoImage = 0;
    }

    m_stereoImage = new StereoImage::StereoImageParameters(s);
    m_HomographyMatrix = H;
}

void ReconstructionThread::bm_arrayToMat(uchar* arr, cv::Mat &image)
{
    for(int i=0; i<m_stereoImage->height; i++)
        for(int j=0; j<m_stereoImage->step; j++)
        { image.at<uchar>(i,j) = *arr, arr++; }
}

void ReconstructionThread::bm_matToArray(cv::Mat image, float* arr)
{
    for(int i=0; i<m_stereoImage->height; i++)
        for(int j=0; j<m_stereoImage->width; j++)
        { *arr = image.at<float>(i,j); arr++; }
}

void ReconstructionThread::sps_arrayToPng(uchar* arr, png::image<png::rgb_pixel> &imagePng)
{
    for(int i=0; i<m_stereoImage->height; i++)
        for(int j=0; j<m_stereoImage->step; j++)
        {
            png::rgb_pixel rgbPixel(*arr, *arr, *arr);
            imagePng.set_pixel(j, i, rgbPixel);
            arr++;
        }
}

void ReconstructionThread::sps_pngToArrary(png::image<png::gray_pixel_16> &imagePng, float* arr)
{
    for(int i=0; i<m_stereoImage->height; i++)
        for(int j=0; j<m_stereoImage->width; j++)
        {   *arr = (imagePng.get_pixel(j, i) / 256.0); arr++;  }
}

void ReconstructionThread::mono_ndarrayToArray(PyObject *Return, float *arr)
{
    this->pReturnArray = (PyArrayObject *)PyList_GetItem(Return, 0);
    int rows = this->pReturnArray->dimensions[0], cols = this->pReturnArray->dimensions[1];
    for( int i = 0; i < rows; i++)
        for( int j = 0; j < cols; j++)
        {   
            int16_t temp = static_cast<int16_t>(*(this->pReturnArray->data + i * this->pReturnArray->strides[0] + j * this->pReturnArray->strides[1]))  ;
            if(temp <0) temp =256 +temp;
            *arr = float(temp * 1.0); arr++;
        }                    
}

void ReconstructionThread::run()
{
    clock_t startTime,endTime;

    //wait for load stereo images;
    while(!StereoImage::waitForReconstruction) usleep(1);

    int32_t d_width  = this->m_stereoImage->width;
    int32_t d_height = this->m_stereoImage->height;
    int32_t d_step   = this->m_stereoImage->step;    

    //free the disparity pointer.
    if(this->m_stereoImage->D1 != 0)
    {    free(this->m_stereoImage->D1); this->m_stereoImage->D1 = 0;  }

    if(this->m_stereoImage->D2 != 0)
    {    free(this->m_stereoImage->D2); this->m_stereoImage->D2 = 0;  }
    
    this->m_stereoImage->D1 = (float*)malloc(d_width*d_height*sizeof(float));
    this->m_stereoImage->D2 = (float*)malloc(d_width*d_height*sizeof(float));

    //compute disparity mode,contain Elas, BM and SGBM.
    //init ELas
    Elas elas(param);
    const int32_t dims[3] = {d_width, d_height, d_step};

    //init SGBM BM
    Mat bm_leftImage( d_height, d_step, CV_8U);
    Mat bm_rightImage( d_height, d_step, CV_8U);

    bm_arrayToMat(this->m_stereoImage->I1, bm_leftImage);
    bm_arrayToMat(this->m_stereoImage->I2, bm_rightImage);

    //init SPS_Stereo
    png::image<png::rgb_pixel> sps_leftImage(d_step, d_height);
    png::image<png::rgb_pixel> sps_rightImage(d_step, d_height);
    png::image<png::gray_pixel_16> segmentImage;
    png::image<png::gray_pixel_16> disparityImage;

    sps_arrayToPng(this->m_stereoImage->I1, sps_leftImage);
    sps_arrayToPng(this->m_stereoImage->I2, sps_rightImage);
    
    //init MonoDepth


 
    ////////////////////
    startTime = clock();

    switch( StereoImage::computeDisparityMode )
    {
        case ELAS_MODE:

            elas.process(m_stereoImage->I1,m_stereoImage->I2,m_stereoImage->D1,m_stereoImage->D2,dims);
            break;

        case SGBM_MODE:
            //sgbm->compute(I1, I2, D1)
            sgbm.setImages(bm_leftImage, bm_rightImage);
            this->bm_matToArray(sgbm.updateSGBM(), m_stereoImage->D1);
            break;

        case BM_MODE:
            //bm->compute(I1, I2, D1)
            bm.setImages(bm_leftImage, bm_rightImage);
            this->bm_matToArray(bm.updateBM(), m_stereoImage->D1);
            break;

        case SPS_MODE:
            //sps.compute(__, left, right, segmentImage, disparityImage);
            sps.compute(superpixelTotal, sps_leftImage, sps_rightImage, segmentImage, disparityImage);
            this->sps_pngToArrary(disparityImage, m_stereoImage->D1);
            break;
        
        case MONO_MODE:
            // setting the --image_path   and run pFunc
            this->pName = PyString_FromString(g_LeftImageName.data());
            PyTuple_SetItem(this->pArg, 0, this->pName);
            this->pReturn = PyEval_CallObject(this->pFunc, this->pArg);
            this->mono_ndarrayToArray(this->pReturn, m_stereoImage->D1);
            break;
    }

    endTime = clock();

    //copy disparity to m_forSaveDepth.
    if( StereoImage::saveToPgm || StereoImage::depthImageShow )
    {
        free( StereoImage::m_forSaveDepth->D1 );
        free( StereoImage::m_forSaveDepth->D2 );
        free( StereoImage::m_forSaveDepth->I1 );
        free( StereoImage::m_forSaveDepth->I2 );

        StereoImage::m_forSaveDepth->width = m_stereoImage->width;
        StereoImage::m_forSaveDepth->height = m_stereoImage->height;
        StereoImage::m_forSaveDepth->step = m_stereoImage->step;
        //depth
        StereoImage::m_forSaveDepth->D1 = (float*)malloc(m_stereoImage->width*m_stereoImage->height*sizeof(float));
        StereoImage::m_forSaveDepth->D2 = (float*)malloc(m_stereoImage->width*m_stereoImage->height*sizeof(float));
        memcpy(StereoImage::m_forSaveDepth->D1, m_stereoImage->D1, m_stereoImage->width*m_stereoImage->height*sizeof(float));
        memcpy(StereoImage::m_forSaveDepth->D2, m_stereoImage->D2, m_stereoImage->width*m_stereoImage->height*sizeof(float));
        //image
        StereoImage::m_forSaveDepth->I1 = (uchar*)malloc(m_stereoImage->step*m_stereoImage->height*sizeof(uchar));
        StereoImage::m_forSaveDepth->I2 = (uchar*)malloc(m_stereoImage->step*m_stereoImage->height*sizeof(uchar));
        memcpy(StereoImage::m_forSaveDepth->I1, m_stereoImage->I1, m_stereoImage->step*m_stereoImage->height*sizeof(uchar));
        memcpy(StereoImage::m_forSaveDepth->I2, m_stereoImage->I2, m_stereoImage->step*m_stereoImage->height*sizeof(uchar));
    }

    static int count = 0;

    cout<< "No."<< ++count << " image depth:  " << (double)(endTime-startTime)/CLOCKS_PER_SEC << "s." <<endl;

    AddDisparityMapToReconstruction();

    StereoImage::waitForReconstruction = false;  //load iamge thread wait
    m_pickedNewMap = false;
    emit DetectNewMap();
    while (!m_pickedNewMap) usleep(1000);
}

ReconstructionThread::Map3d ReconstructionThread::CreateCurrentMap()
{
    Map3d m;
    m.I      = (float*)malloc(m_stereoImage->width*m_stereoImage->height*sizeof(float));
    m.D      = (float*)malloc(m_stereoImage->width*m_stereoImage->height*sizeof(float));
    m.X      = (float*)malloc(m_stereoImage->width*m_stereoImage->height*sizeof(float));
    m.Y      = (float*)malloc(m_stereoImage->width*m_stereoImage->height*sizeof(float));
    m.Z      = (float*)malloc(m_stereoImage->width*m_stereoImage->height*sizeof(float));
    m.H      = m_HomographyMatrix;
    m.width  = m_stereoImage->width;
    m.height = m_stereoImage->height;

    // copy disparities and intensities
    memcpy(m.D,m_stereoImage->D1,m_stereoImage->width*m_stereoImage->height*sizeof(float));
    for (int32_t v=0; v<m.height; v++)
    {
        for (int32_t u=0; u<m.width; u++)
        {
            m.I[v*m.width+u] = ((float)m_stereoImage->I1[v*m_stereoImage->step+u])/255.0;
        }
    }

    // compute 3d coordinates in first coordinate system
    float hcf00=0,hcf01=0,hcf02=0,hcf03=0,hcf10=0,hcf11=0,hcf12=0,hcf13=0,hcf20=0,hcf21=0,hcf22=0,hcf23=0;

    hcf00 = m.H._val[0][0]; hcf01 = m.H._val[0][1]; hcf02 = m.H._val[0][2]; hcf03 = m.H._val[0][3];
    hcf10 = m.H._val[1][0]; hcf11 = m.H._val[1][1]; hcf12 = m.H._val[1][2]; hcf13 = m.H._val[1][3];
    hcf20 = m.H._val[2][0]; hcf21 = m.H._val[2][1]; hcf22 = m.H._val[2][2]; hcf23 = m.H._val[2][3];

    for (int32_t u=0; u<m.width; u++)
    {
        for (int32_t v=0; v<m.height; v++)
        {
            int32_t addr = v*m.width+u;
            float   d    = m.D[addr];
            if (d>0)
            {
                float z = (m_f * m_baseline)/d;
                if ((z>m_showingMinDistance) && (z < m_showingMaxDistance))
                {
                    float x = ((float)u - m_cu)*m_baseline/d;
                    float y = ((float)v - m_cv)*m_baseline/d;
                    m.X[addr] = hcf00*x+hcf01*y+hcf02*z+hcf03;
                    m.Y[addr] = hcf10*x+hcf11*y+hcf12*z+hcf13;
                    m.Z[addr] = hcf20*x+hcf21*y+hcf22*z+hcf23;
                }
                else
                {
                    m.D[addr] = -1;
                }
            }
        }
    }

    return m;
}

void ReconstructionThread::AddDisparityMapToReconstruction()
{

    Map3d currentMap3d = CreateCurrentMap();

    m_mapPoints.clear();

    // for all pixels do
    for (int32_t u = 0; u < currentMap3d.width; u++)
    {
        for (int32_t v = 0; v < currentMap3d.height; v++)
        {
            // get pixel address and disparity
            int32_t addr = v * currentMap3d.width + u;
            float   d    = currentMap3d.D[addr];

            // if disparity is valid: add 3d coordinates to point cloud
            if (d>0)
            {
                m_mapPoints.push_back( ModelPainting::Point3D(currentMap3d.X[addr],
                                                              currentMap3d.Y[addr],
                                                              currentMap3d.Z[addr],
                                                              currentMap3d.I[addr]) );
            }
        }
    }

    ReleaseMap(currentMap3d);
}

void ReconstructionThread::ReleaseMap(ReconstructionThread::Map3d &m)
{
    if(m.I != 0)
    {
        free(m.I);
        m.I = 0;
    }
    if(m.D != 0)
    {
        free(m.D);
        m.D = 0;
    }
    if(m.X != 0)
    {
        free(m.X);
        m.X = 0;
    }
    if(m.Y != 0)
    {
        free(m.Y);
        m.Y = 0;
    }
    if(m.Z != 0)
    {
        free(m.Z);
        m.Z = 0;
    }
}
