# seam-carving
Implementation of seam carving algorithm following the paper 'Seam Carving for Content-Aware Image Resizing' by Avidan and Shamir.

https://perso.crans.org/frenoy/matlab2012/seamcarving.pdf

results of running this on broadway tower
![download (4)](https://github.com/user-attachments/assets/00ee6a46-7a5c-4927-b0f6-d6f9bca5d842)

watch my yt video on object removal:
[https://studio.youtube.com/video/UZG52d_SKC0/edit](https://www.youtube.com/watch?v=UZG52d_SKC0)

- [x] energy matrix
- [x] seam finding (vertical)
- [x] seam finding (horizontal)
- [x] draw seam
- [x] remove seam
- [x] resize based on desired width and height
- [x] object removal
- [x] object protection
- [x] seam insertion
- [x] gui app for demo purposes.
- [ ] see if i can make like autogenerated gifs of seam carving process (done-ish, got an ffmpeg script)
- [ ] separate rendering and logic portions for cleanness
- [ ] speed optimization (partially done)
- [ ] performance evaluation
- [ ] make this into a library
- [ ] resize to any size (i.e. one axis needs to have added seams, the other removed seams)
- [ ] allow users to pass in their own energy function

Project limitations:
1. Implementation is limited to the original paper, not the later published paper
2. Energy function used is dual gradient energy function, not entrophy or hog.
3. No transport map for optimized seam removal order.
4. No poisson solver.

Demo results:

|Item            | Before| After  |
|-------------   |-------|--------|
| Seam Removal   | ![before](https://i.ibb.co/SKJs7kW/lincoln-park.jpg)| ![after](https://i.ibb.co/BgTYM8z/lincoln-park-new.png)|
| Object Removal | ![before](https://i.ibb.co/PG1LZ45/2peng.jpg)       | ![after](https://i.ibb.co/NSz7nB9/lincoln-park-new.png)       |
| Seam Insertion | ![before](https://i.ibb.co/0Z27fkG/zen-garden1.jpg)       | ![after](https://i.ibb.co/dMWG9gW/zengarden.png)       |