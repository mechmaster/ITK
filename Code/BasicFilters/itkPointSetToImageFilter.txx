/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    itkPointSetToImageFilter.txx
Language:  C++
Date:      $Date$
Version:   $Revision$

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkPointSetToImageFilter_txx
#define _itkPointSetToImageFilter_txx

#include "itkPointSetToImageFilter.h"
#include <itkImageRegionIteratorWithIndex.h>

namespace itk
{

/** Constructor */
template <class TInputPointSet, class TOutputImage>
PointSetToImageFilter<TInputPointSet,TOutputImage>
::PointSetToImageFilter()
{
  this->SetNumberOfRequiredInputs(1);
  m_Size.Fill(0);
  
  for (unsigned int i = 0; i < OutputImageDimension; i++)
    {
    m_Spacing[i] = 1.0;
    m_Origin[i] = 0;
    }

  m_InsideValue = 1;
  m_OutsideValue = 0;
}

/** Destructor */
template <class TInputPointSet, class TOutputImage>
PointSetToImageFilter<TInputPointSet,TOutputImage>
::~PointSetToImageFilter()
{
}
  

/** Set the Input PointSet */
template <class TInputPointSet, class TOutputImage>
void 
PointSetToImageFilter<TInputPointSet,TOutputImage>
::SetInput(const InputPointSetType *input)
{
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(0, 
                                   const_cast< InputPointSetType * >( input ) );
}


/** Connect one of the operands  */
template <class TInputPointSet, class TOutputImage>
void
PointSetToImageFilter<TInputPointSet,TOutputImage>
::SetInput( unsigned int index, const TInputPointSet * pointset ) 
{
  if( index+1 > this->GetNumberOfInputs() )
    {
    this->SetNumberOfRequiredInputs( index + 1 );
    }
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(index, 
                                   const_cast< TInputPointSet *>( pointset ) );
}



/** Get the input point-set */
template <class TInputPointSet, class TOutputImage>
const typename PointSetToImageFilter<TInputPointSet,TOutputImage>::InputPointSetType *
PointSetToImageFilter<TInputPointSet,TOutputImage>
::GetInput(void) 
{
  if (this->GetNumberOfInputs() < 1)
    {
    return 0;
    }
  
  return static_cast<const TInputPointSet * >
    (this->ProcessObject::GetInput(0) );
}
  
/** Get the input point-set */
template <class TInputPointSet, class TOutputImage>
const typename PointSetToImageFilter<TInputPointSet,TOutputImage>::InputPointSetType *
PointSetToImageFilter<TInputPointSet,TOutputImage>
::GetInput(unsigned int idx)
{
  return static_cast< const TInputPointSet * >
    (this->ProcessObject::GetInput(idx));
}

//----------------------------------------------------------------------------
template <class TInputPointSet, class TOutputImage>
void 
PointSetToImageFilter<TInputPointSet,TOutputImage>
::SetSpacing(const double spacing[OutputImageDimension] )
{
  unsigned int i; 
  for (i=0; i<OutputImageDimension; i++)
    {
    if ( spacing[i] != m_Spacing[i] )
      {
      break;
      }
    } 
  if ( i < OutputImageDimension ) 
    { 
    for (i=0; i<OutputImageDimension; i++)
      {
      m_Spacing[i] = spacing[i];
      }
    }
}

template <class TInputPointSet, class TOutputImage>
void 
PointSetToImageFilter<TInputPointSet,TOutputImage>
::SetSpacing(const float spacing[OutputImageDimension] )
{
  unsigned int i; 
  for (i=0; i<OutputImageDimension; i++)
    {
    if ( (double)spacing[i] != m_Spacing[i] )
      {
      break;
      }
    } 
  if ( i < OutputImageDimension ) 
    { 
    for (i=0; i<OutputImageDimension; i++)
      {
      m_Spacing[i] = spacing[i];
      }
    }
}

template <class TInputPointSet, class TOutputImage>
const double * 
PointSetToImageFilter<TInputPointSet,TOutputImage>
::GetSpacing() const
{
  return m_Spacing;
}

//----------------------------------------------------------------------------
template <class TInputPointSet, class TOutputImage>
void 
PointSetToImageFilter<TInputPointSet,TOutputImage>
::SetOrigin(const double origin[OutputImageDimension] )
{
  unsigned int i; 
  for (i=0; i<OutputImageDimension; i++)
    {
    if ( origin[i] != m_Origin[i] )
      {
      break;
      }
    } 
  if ( i < OutputImageDimension ) 
    { 
    for (i=0; i<OutputImageDimension; i++)
      {
      m_Origin[i] = origin[i];
      }
    }
}

template <class TInputPointSet, class TOutputImage>
void 
PointSetToImageFilter<TInputPointSet,TOutputImage>
::SetOrigin(const float origin[OutputImageDimension] )
{
  unsigned int i; 
  for (i=0; i<OutputImageDimension; i++)
    {
    if ( (double)origin[i] != m_Origin[i] )
      {
      break;
      }
    } 
  if ( i < OutputImageDimension ) 
    { 
    for (i=0; i<OutputImageDimension; i++)
      {
      m_Origin[i] = origin[i];
      }
    }
}

template <class TInputPointSet, class TOutputImage>
const double * 
PointSetToImageFilter<TInputPointSet,TOutputImage>
::GetOrigin() const
{
  return m_Origin;
}

//----------------------------------------------------------------------------

/** Update */
template <class TInputPointSet, class TOutputImage>
void
PointSetToImageFilter<TInputPointSet,TOutputImage>
::GenerateData(void)
{
  unsigned int i;
  itkDebugMacro(<< "PointSetToImageFilter::Update() called");

  // Get the input and output pointers 
  const InputPointSetType * InputPointSet  = this->GetInput();
  OutputImagePointer   OutputImage = this->GetOutput();

  // Generate the image
  double origin[InputPointSetDimension];
  SizeType size;

  const InputPointSetType::BoundingBoxType* bb = InputPointSet->GetBoundingBox();

  for(i=0;i<InputPointSetDimension;i++)
    {
    size[i] = bb->GetBounds()[2*i+1]-bb->GetBounds()[2*i];
    origin[i]=0;
    }
  

  typename OutputImageType::RegionType region;
  
  // If the size of the output has been explicitly specified, the filter
  // will set the output size to the explicit size, otherwise the size from the spatial
  // PointSet's bounding box will be used as default.

  bool specified = false;
  for (i = 0; i < OutputImageDimension; i++)
    {
    if (m_Size[i] != 0)
      {
      specified = true;
      break;
      }
    }

  if (specified)
    {
    region.SetSize( m_Size );
    }
  else
    {
    region.SetSize( size );
    }

  OutputImage->SetRegions( region);
  
  // If the spacing has been explicitly specified, the filter
  // will set the output spacing to that explicit spacing, otherwise the spacing from
  // the point-set is used as default.
  
  specified = false;
  for (i = 0; i < OutputImageDimension; i++)
    {
    if (m_Spacing[i] != 0)
      {
      specified = true;
      break;
      }
    }

  if (specified)
    {
    OutputImage->SetSpacing(this->m_Spacing);         // set spacing
    }

  OutputImage->SetOrigin(origin);   //   and origin
  OutputImage->Allocate();   // allocate the image   
  OutputImage->FillBuffer(m_OutsideValue);

  typedef typename InputPointSetType::PointsContainer::ConstIterator  PointIterator;
  PointIterator pointItr = InputPointSet->GetPoints()->Begin();
  PointIterator pointEnd = InputPointSet->GetPoints()->End();

  typename OutputImageType::IndexType index;

  while( pointItr != pointEnd )
    {
    if(OutputImage->TransformPhysicalPointToIndex(pointItr.Value(),index))
      {
      OutputImage->SetPixel(index,m_InsideValue);
      }
    pointItr++;
    }
  
  itkDebugMacro(<< "PointSetToImageFilter::Update() finished");

} // end update function  


template<class TInputPointSet, class TOutputImage>
void 
PointSetToImageFilter<TInputPointSet,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  unsigned int i=0;
  Superclass::PrintSelf(os, indent);
  os << indent << "Size : " << m_Size << std::endl;
  std::cout << "Origin : " ;  
  for(i=0;i<InputPointSetDimension;i++)
    {
    std::cout << m_Origin[i] << " ";
    }
  std::cout << std::endl;
  std::cout << "Spacing : " ;  
  for(i=0;i<InputPointSetDimension;i++)
    {
    std::cout << m_Spacing[i] << " ";
    }
  std::cout << std::endl;
  os << indent << "Inside Value : " << m_InsideValue << std::endl;
  os << indent << "Outside Value : " << m_OutsideValue << std::endl;
}



} // end namespace itk

#endif
