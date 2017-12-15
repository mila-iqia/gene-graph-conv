#Synthetic data

## Running script
1. To generate the dataset with the 10x10 graph and 100 examples run the script:
*python main.py --size_x 10 --size_y 10 --N 100 --dataset test.hdf5*

2. To generate some examples with visualization:
*python main.py --test 1*
here is the example of output:
<table style="width:100%">
  <tr>
    <th>Percolates</th>
    <th>Does not percolate</th>
  </tr>
  <tr>
    <th>
      <img src="https://raw.githubusercontent.com/ieee8023/conv-graph/master/Dataset/SyntheticData/Fig/Example_1.png" width="400">
    </th>
    <th>
      <img src="https://raw.githubusercontent.com/ieee8023/conv-graph/master/Dataset/SyntheticData/Fig/Example_2.png" width="400">
    </th>
  </tr>
</table>

3. To inspect the example 1 the dataset *test.hdf5*:
*python main.py --test 1 --dataset test.hdf5*