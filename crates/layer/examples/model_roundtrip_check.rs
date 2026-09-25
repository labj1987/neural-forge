//! Hardware release check. Requires the candidate helper on the same NEURAL_FORGE_SHM,
//! a gameplay PNG argument, and an output directory argument. Never use the live SHM.
//! Motion vectors, if enabled, are estimated by the helper itself from consecutive
//! proxies (frame 1 of each pair is a shifted copy of frame 0 to give it real motion).
use std::{fs::File, os::unix::fs::FileExt, sync::atomic::Ordering, time::{Duration,Instant}};
use neural_forge_protocol::{enums::proxy_format, mapping};
fn save(path: &std::path::Path, pixels:&[u8],w:u32,h:u32) {
    let mut enc=png::Encoder::new(File::create(path).unwrap(),w,h);
    enc.set_color(png::ColorType::Rgba);enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(pixels).unwrap();
}
fn main() {
    let args:Vec<_>=std::env::args().collect();
    let out=std::path::Path::new(&args[2]);std::fs::create_dir_all(out).unwrap();
    let mut reader=png::Decoder::new(File::open(&args[1]).unwrap()).read_info().unwrap();
    let mut all=vec![0;reader.output_buffer_size()];let info=reader.next_frame(&mut all).unwrap();
    assert_eq!(info.bit_depth,png::BitDepth::Eight);
    let channels=match info.color_type {png::ColorType::Rgb=>3,png::ColorType::Rgba=>4,_=>panic!("RGB PNG required")};
    let (w,h)=(512u32,512u32);assert!(info.width>=w && info.height>=h);
    let (ox,oy)=((info.width-w)/2,(info.height-h)/2);
    let mut pixels=vec![255u8;(w*h*4) as usize];
    for y in 0..h {for x in 0..w {let src=(((y+oy)*info.width+x+ox)*channels) as usize;let dst=((y*w+x)*4) as usize;pixels[dst..dst+3].copy_from_slice(&all[src..src+3]);}}
    save(&out.join("input.png"),&pixels,w,h);
    let m=mapping::open().unwrap();let hdr=m.header();
    let file=std::fs::OpenOptions::new().read(true).write(true).open(neural_forge_protocol::env::var("NEURAL_FORGE_SHM").unwrap()).unwrap();
    let deadline=Instant::now()+Duration::from_secs(30);
    while hdr.helper_state.load(Ordering::Acquire)!=neural_forge_protocol::enums::helper_state::RUNNING {
        assert!(Instant::now()<deadline,"helper did not start");std::thread::sleep(Duration::from_millis(25));
    }
    let mut outputs=vec![];
    for (i,format) in [proxy_format::RGBA8,proxy_format::BGRA8].into_iter().enumerate() {
        for frame in 0..2 {
            let mut input=pixels.clone();
            if frame==1 {for y in 0..h {for x in 8..w {let dst=((y*w+x)*4) as usize;input[dst..dst+4].copy_from_slice(&pixels[dst-32..dst-28]);}}}
            if format==proxy_format::BGRA8 {for p in input.as_chunks_mut::<4>().0 {p.swap(0,2);}}
            file.write_all_at(&input,neural_forge_protocol::proxy_offset() as u64).unwrap();
            hdr.width.store(w,Ordering::Relaxed);hdr.height.store(h,Ordering::Relaxed);hdr.proxy_format.store(format,Ordering::Relaxed);
            let req=hdr.seq_req.load(Ordering::Relaxed)+1;let start=Instant::now();hdr.seq_req.store(req,Ordering::Release);
            while hdr.seq_resp.load(Ordering::Acquire)!=req {assert!(start.elapsed()<Duration::from_secs(30),"request timed out");std::thread::sleep(Duration::from_millis(2));}
            assert_eq!(hdr.model_up.load(Ordering::Relaxed),1,"model not available");
            let mut answer=vec![0u8;input.len()];file.read_exact_at(&mut answer,neural_forge_protocol::answer_offset() as u64).unwrap();
            if format==proxy_format::BGRA8 {for p in answer.as_chunks_mut::<4>().0 {p.swap(0,2);}}
            println!("format={format} frame={frame} response={:?}",start.elapsed());
            save(&out.join(format!("output-{i}-{frame}.png")),&answer,w,h);
            if frame==0 {outputs.push(answer);}
        }
    }
    hdr.quit.store(1,Ordering::Release);
    let diff=outputs[0].iter().zip(&outputs[1]).map(|(a,b)|a.abs_diff(*b) as u64).sum::<u64>() as f64 / outputs[0].len() as f64;
    println!("RGBA/BGRA semantic mean absolute difference = {diff}");
    assert!(diff<3.0,"RGBA/BGRA representations must produce comparable semantic colors");
}
