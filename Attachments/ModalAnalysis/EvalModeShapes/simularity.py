def simularity(data, ref):

          
        
          data = np.array(data)
          ref = np.array(ref)
          n = len(data)
          rev_refdata = ref[::-1]
          
          RMS = np.sqrt(np.sum((ref-data)**2)/n)
          RMS_rev = np.sqrt(np.sum((rev_refdata-data)**2)/n)
          
          RMS = min(RMS,RMS_rev)
          
          simularity = round(abs(100-(RMS*100)),1)
          
          return simularity