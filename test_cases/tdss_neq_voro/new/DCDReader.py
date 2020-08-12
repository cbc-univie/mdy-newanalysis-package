class DCDReader(base.Reader):
    """Reads from a DCD file

    :Data:
        ts
          :class:`~MDAnalysis.coordinates.base.Timestep` object
          containing coordinates of current frame

    :Methods:
        ``dcd = DCD(dcdfilename)``
           open dcd file and read header
        ``len(dcd)``
           return number of frames in dcd
        ``for ts in dcd:``
           iterate through trajectory
        ``for ts in dcd[start:stop:skip]:``
           iterate through a trajectory
        ``dcd[i]``
           random access into the trajectory (i corresponds to frame number)
        ``data = dcd.timeseries(...)``
           retrieve a subset of coordinate information for a group of atoms
        ``data = dcd.correl(...)``
           populate a :class:`MDAnalysis.core.Timeseries.Collection` object with computed timeseries
    """
    format = 'DCD'
    units = {'time': 'AKMA', 'length': 'Angstrom'}

    def __init__(self, dcdfilename, **kwargs):
        self.dcdfilename = dcdfilename
        self.filename = self.dcdfilename
        self.dcdfile = None  # set right away because __del__ checks

        # Issue #32: segfault if dcd is 0-size
        # Hack : test here... (but should be fixed in dcd.c)
        stats = os.stat(self.dcdfilename)
        if stats.st_size == 0:
            raise IOError(errno.EIO,"DCD file is zero size",dcdfilename)

        #self.dcdfile = open(dcdfilename, 'rb')
        self.numatoms = 0
        self.numframes = 0
        self.fixed = 0
        self.skip = 1
        self.periodic = False

        self._read_dcd_header()
        self.ts = Timestep(self.numatoms)
        # Read in the first timestep
        self._read_next_timestep()
    def _dcd_header(self):
        """Returns contents of the DCD header C structure::
             typedef struct {
               fio_fd fd;                 // FILE *
               fio_size_t header_size;    // size_t == sizeof(int)
               int natoms;
               int nsets;
               int setsread;
               int istart;
               int nsavc;
               double delta;
               int nfixed;
               int *freeind;
               float *fixedcoords;
               int reverse;
               int charmm;
               int first;
               int with_unitcell;
             } dcdhandle;

        .. deprecated:: 0.7.5
           This function only exists for debugging purposes and might
           be removed without notice. Do not rely on it.

        """
        # was broken (no idea why [orbeckst]), see Issue 27
        # 'PiiiiiidiPPiiii' should be the unpack string according to the struct.
        #    struct.unpack("LLiiiiidiPPiiii",self._dcd_C_str)
        # seems to do the job on Mac OS X 10.6.4 ... but I have no idea why,
        # given that the C code seems to define them as normal integers
        import struct
        desc = ['file_desc', 'header_size', 'natoms', 'nsets', 'setsread', 'istart', 'nsavc', 'delta', 'nfixed', 'freeind_ptr', 'fixedcoords_ptr', 'reverse', 'charmm', 'first', 'with_unitcell']
        return dict(list(zip(desc, struct.unpack("LLiiiiidiPPiiii",self._dcd_C_str))))
    def __iter__(self):
        # Reset the trajectory file, read from the start
        # usage is "from ts in dcd:" where dcd does not have indexes
        self._reset_dcd_read()
        def iterDCD():
            for i in range(0, self.numframes, self.skip):  # FIXME: skip is not working!!!
                try: yield self._read_next_timestep()
                except IOError: raise StopIteration
        return iterDCD()
    def _read_next_timestep(self, ts=None):
        if ts is None:
            ts = self.ts

        ts.frame = self._read_next_frame(ts._x, ts._y, ts._z, ts._unitcell, self.skip)

        return ts
    def __getitem__(self, frame):
        if (numpy.dtype(type(frame)) != numpy.dtype(int)) and (type(frame) != slice):
            raise TypeError
        if (numpy.dtype(type(frame)) == numpy.dtype(int)):
            if (frame < 0):
                # Interpret similar to a sequence
                frame = len(self) + frame
            if (frame < 0) or (frame >= len(self)):
                raise IndexError
            self._jump_to_frame(frame)  # XXX required!!
            ts = self.ts
            ts.frame = self._read_next_frame(ts._x, ts._y, ts._z, ts._unitcell, 1) # XXX required!!
            return ts
        elif type(frame) == slice: # if frame is a slice object
            if not (((type(frame.start) == int) or (frame.start == None)) and
                    ((type(frame.stop) == int) or (frame.stop == None)) and
                    ((type(frame.step) == int) or (frame.step == None))):
                raise TypeError("Slice indices are not integers")
            def iterDCD(start=frame.start, stop=frame.stop, step=frame.step):
                start, stop, step = self._check_slice_indices(start, stop, step)
                for i in range(start, stop, step):
                    yield self[i]
            return iterDCD()
    def close(self):
        self._finish_dcd_read()
        self.dcdfile = None
    def Writer(self, filename, **kwargs):
        """Returns a DCDWriter for *filename* with the same parameters as this DCD.

        All values can be changed through keyword arguments.

        :Arguments:
          *filename*
              filename of the output DCD trajectory
        :Keywords:
          *numatoms*
              number of atoms
          *start*
              number of the first recorded MD step
          *step*
              indicate that *step* MD steps (!) make up one trajectory frame
          *delta*
              MD integrator time step (!), in AKMA units
          *remarks*
              string that is stored in the DCD header [XXX -- max length?]

        :Returns: :class:`DCDWriter`

        .. Note::

           The keyword arguments set the low-level attributes of the DCD
           according to the CHARMM format. The time between two frames would be
           *delta* * *step* !
        """
        numatoms = kwargs.pop('numatoms', self.numatoms)
        kwargs.setdefault('start', self.start_timestep)
        kwargs.setdefault('step', self.skip_timestep)
        kwargs.setdefault('delta', self.delta)
        kwargs.setdefault('remarks', self.remarks)
        return DCDWriter(filename, numatoms, **kwargs)
    def __del__(self):
        if not self.dcdfile is None:
            self.close()

class Timestep(object):
    """Timestep data for one frame

    :Methods:

      ``ts = Timestep(numatoms)``

         create a timestep object with space for numatoms (done
         automatically)

      ``ts[i]``

         return coordinates for the i'th atom (0-based)

      ``ts[start:stop:skip]``

         return an array of coordinates, where start, stop and skip
         correspond to atom indices,
         :attr:`MDAnalysis.core.AtomGroup.Atom.number` (0-based)

      ``for x in ts``

         iterate of the coordinates, atom by atom
    """
    def __init__(self, arg, **kwargs):
        if numpy.dtype(type(arg)) == numpy.dtype(int):
            self.frame = 0
            self.numatoms = arg
            self._pos = numpy.zeros((self.numatoms, 3), dtype=numpy.float32, order='F')
            #self._pos = numpy.zeros((3, self.numatoms), numpy.float32)
            self._unitcell = numpy.zeros((6), numpy.float32)
        elif isinstance(arg, Timestep): # Copy constructor
            # This makes a deepcopy of the timestep
            self.frame = arg.frame
            self.numatoms = arg.numatoms
            self._unitcell = numpy.array(arg._unitcell)
            self._pos = numpy.array(arg._pos, order='F')
        elif isinstance(arg, numpy.ndarray): # Init using a 3N coordinate array
            if len(arg.shape) != 2:
                raise ValueError("numpy array can only have 2 dimensions")
            self._unitcell = numpy.zeros((6), numpy.float32)
            self.frame = 0
            if arg.shape[1] == 3:
                self.numatoms = arg.shape[0]
            else:
                self.numatoms = arg.shape[0]
                # Or should an exception be raised if coordinate
                # structure is not 3-dimensional? Maybe velocities
                # could be read one day... [DP]
            self._pos = arg.astype(numpy.float32).copy('Fortran',)
        else:
            raise ValueError("Cannot create an empty Timestep")
        self._x = self._pos[:,0]
        self._y = self._pos[:,1]
        self._z = self._pos[:,2]
