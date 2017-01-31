/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include <assert.h>
#include "itkGPUContextManager.h"

namespace itk
{
// static variable initialization
GPUContextManager* GPUContextManager::m_Instance = ITK_NULLPTR;

GPUContextManager* GPUContextManager::GetInstance()
{
  if(m_Instance == ITK_NULLPTR)
    {
    m_Instance = new GPUContextManager();
    }
  return m_Instance;
}

void GPUContextManager::DestroyInstance()
{
  m_Instance->Delete();
  m_Instance = ITK_NULLPTR;
  itkDebugStatement(std::cout << "OpenCL context is destroyed." << std::endl);
}

GPUContextManager::GPUContextManager()
{
  cl_int errid;

  m_DevicesList.clear();
  
  std::vector<cl_platform_id> platformsList = OpenCLGetPlatformsList();
  if (platformsList.size() != 0)
    {
    cl_device_type devType = CL_DEVICE_TYPE_GPU;
    
    std::vector<cl_platform_id>::iterator iter = platformsList.begin();
    for (; iter != platformsList.end(); ++iter)
      {
      std::vector<cl_device_id> devicesList;
      bool res = OpenCLGetAvailableDevices(*iter, devType, devicesList);
      if (res && devicesList.size())
        {
          m_DevicesList.insert(m_DevicesList.end(), devicesList.begin(), devicesList.end());
        }
      }
    }
  
  if (m_DevicesList.size() == 0)
    {
    errid = CL_DEVICE_NOT_AVAILABLE;
    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
    }
  
  // create context
  m_Context = clCreateContext(ITK_NULLPTR, m_DevicesList.size(), &m_DevicesList[0], ITK_NULLPTR, ITK_NULLPTR, &errid);
  OpenCLCheckError( errid, __FILE__, __LINE__, ITK_LOCATION );

  // create command queues
  m_CommandQueue = (cl_command_queue *)malloc(m_DevicesList.size() * sizeof(cl_command_queue) );
  for(unsigned int i=0; i<m_DevicesList.size(); i++)
    {
      m_CommandQueue[i] = clCreateCommandQueue(m_Context, m_DevicesList[i], 0, &errid);

    // Debug
      OpenCLPrintDeviceInfo(m_DevicesList[i], true);
    //
      OpenCLCheckError( errid, __FILE__, __LINE__, ITK_LOCATION );
    }
}

GPUContextManager::~GPUContextManager()
{
  cl_int errid;
  for(unsigned int i=0; i<m_DevicesList.size(); i++)
    {
    errid = clReleaseCommandQueue(m_CommandQueue[i]);
    OpenCLCheckError( errid, __FILE__, __LINE__, ITK_LOCATION );
    }
  free(m_CommandQueue);
  errid = clReleaseContext(m_Context);
  OpenCLCheckError( errid, __FILE__, __LINE__, ITK_LOCATION );
  
  if (m_DevicesList.size())
    {
      m_DevicesList.clear();
    }
}

cl_command_queue GPUContextManager::GetCommandQueue(int i)
{
  if(i < 0 || i >= (int)m_DevicesList.size())
    {
    printf("Error: requested queue id is not available. Default queue will be used (queue id = 0)\n");
    return m_CommandQueue[0];
    }

  return m_CommandQueue[i];
}

cl_device_id GPUContextManager::GetDeviceId(int i)
{
  if(i < 0 || i >= (int)m_DevicesList.size())
    {
    printf("Error: requested queue id is not available. Default queue will be used (queue id = 0)\n");
  return m_DevicesList[0];
    }

  return m_DevicesList[i];
}

} // namespace itk
