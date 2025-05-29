// GPUGuruDlg.h : header file

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

// CGPUGuruDlg dialog
class CGPUGuruDlg : public CDialogEx
{
public:
	CGPUGuruDlg(CWnd* pParent = nullptr);

	enum FilterType {
		FILTER_NONE,
		FILTER_EDGE,
		FILTER_BLUR,
		FILTER_INVERT
	};

	enum ProcessingMode {
		MODE_CV_ONLY = 0,
		MODE_GPU_ONLY,
		MODE_CV_WITH_GPU
	};


	// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_GPUGURU_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);
	virtual BOOL OnInitDialog();
	//afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	//afx_msg void OnPaint();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	afx_msg void OnBnClickedCbGpu();
	afx_msg void OnCbnSelchangeCbFilter();

	DECLARE_MESSAGE_MAP()

private:
	// UI
	HICON m_hIcon;
	CStatic m_imageDisplay;
	//BOOL m_useGPU = FALSE;
    //ProcessingMode m_processingMode = MODE_CV_ONLY;
	int m_processingMode = MODE_CV_ONLY;

	// Camera and image
	cv::VideoCapture cap;
	cv::Mat frame;

	// Filter selection
	FilterType m_filterType = FILTER_NONE;

	// Timing metrics
	int frameCount = 0;
	double totalTime = 0.0;
	const int maxFrames = 60;

	// Helpers
	void ResizeDialogToFitCamera();
	void UpdateTitleWithMetrics(double avgTime);
	cv::Mat ProcessFrame(const cv::Mat& frame);
	cv::Mat ApplyCpuBlur(const cv::Mat& input);
	cv::Mat ApplyCpuFilterEdge(const cv::Mat& input);
	cv::Mat ApplyCpuInvert(const cv::Mat& input);
public:
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedCancel();
	afx_msg void OnBnClickedStop();
};
