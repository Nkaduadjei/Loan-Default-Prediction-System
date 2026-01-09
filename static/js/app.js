document.addEventListener("DOMContentLoaded", function(){
    const steps = Array.from(document.querySelectorAll(".step"));
    const formSteps = Array.from(document.querySelectorAll(".form-step"));
    let current = 0;
  
    function go(stepIndex){
      formSteps.forEach((s,i) => {
        s.classList.toggle("form-step-active", i===stepIndex);
        steps[i].classList.toggle("active", i===stepIndex);
      });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  
    document.querySelectorAll(".next").forEach(btn=>{
      btn.addEventListener("click", ()=>{
        if(current < formSteps.length-1) current++;
        go(current);
      });
    });
    document.querySelectorAll(".prev").forEach(btn=>{
      btn.addEventListener("click", ()=>{
        if(current > 0) current--;
        go(current);
      });
    });
  
    // initialize
    go(0);
  });
  