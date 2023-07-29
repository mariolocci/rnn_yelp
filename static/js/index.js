var $TABLE = $('#table');
var $BTN = $('#export-btn');
var $TRAINBTN = $('#train-btn');
var $RESTARTBTN = $('#restart-btn');
var $BTNGET = $('.table-add');
var $EXPORT = $('#export');
var restaurant_categories=["FOOD","AMBIENCE","STAFF","SERVICE","FOODVARIETY","FOODDESSERT","FOODBEVERAGE","PRICE","NONE"]
var sentiment=["+","=","-"]
var tmp_objects=[]


function addreview4(data){
  
  var $clone = $TABLE.find('tr.hide').clone(true).removeClass('hide table-line');
  $clone.find('#name').text(data.result.name)
 // $clone.find('#text').text(data.result.text)
  tmp_objects[data.result.name]=data.result
  var tab = document.createElement('table')
   var sent=$clone.find('#sentences')
    sent.append(tab)
 for (s in data.result.sentences){
   tr=newLine(data.result.sentences[s][0],data.result.sentences[s][1],data.result.sentences[s][2],data.result.sentences[s][3],data.result.sentences[s][4],data.result.sentences[s][5],s)
   tr.review=data.result
    tab.append(tr)
   }
   
  $TABLE.find('.table').append($clone);
}


function newLine(text,traslated,label1,label2,label3,sentiment,index) {
 var td=document.createElement('tr')
td.className="nested"

 var tx=document.createElement("td")
 tx.innerHTML=text
 td.append(tx)
 
 var tt=document.createElement("td")
 tt.innerHTML=traslated
 td.append(tt)

 var tl1=document.createElement("td")
 tl1.innerHTML=label1
 tl1.id="labelled"
 tl1.label_index=2
 tl1.sentences_index=index
 td.append(tl1)
 
 var tl2=document.createElement("td")
 tl2.innerHTML=label2
 tl2.id="labelled"
 tl2.sentences_index=index
 tl2.label_index=3
 td.append(tl2)


var tl3=document.createElement("td")
 tl3.innerHTML=label3
 tl3.id="labelled"
 tl3.sentences_index=index
 tl3.label_index=4
 td.append(tl3)

var tl4=document.createElement("td")
 tl4.innerHTML=sentiment
 tl4.id="labelled"
 tl4.sentences_index=index
 tl4.label_index=5
 td.append(tl4)
 return td
  

}


$('.table-adde').click(function () {
  var $clone = $TABLE.find('tr.hide').clone(true).removeClass('hide table-line');
  $TABLE.find('table').append($clone);
});

$('.table-remove').click(function () {
  $(this).parents('tr').detach();
});

$('.table-up').click(function () {
  var $row = $(this).parents('tr');
  if ($row.index() === 1) return; // Don't go above the header
  $row.prev().before($row.get(0));
});

$('.table-down').click(function () {
  var $row = $(this).parents('tr');
  $row.next().after($row.get(0));
});


$('td').on("click","#labelled",function () {
  var $colon = $(this);
  var categories
  if(this.label_index>4)
       categories=sentiment
  else 
      categories=restaurant_categories
  idx= categories.indexOf($colon.text());
  
  if (idx<categories.length-1)
      $colon.text(categories[idx+1])
  else
      $colon.text(categories[0])
  var $row = $(this).parent();
  var sentences_index=this.sentences_index
  var current_review=$row[0].review
  current_review.sentences[sentences_index][this.label_index]=$colon.text()
  current_review.classified=true
  postData("reviews/update",current_review)

});

// A few jQuery helpers for exporting only
jQuery.fn.pop = [].pop;
jQuery.fn.shift = [].shift;

$BTN.click(function () {
  var $rows = $TABLE.find('tr:not(:hidden)');
  var headers = [];
  var data = [];
  
  // Get the headers (add special header logic here)
  $($rows.shift()).find('th:not(:empty)').each(function () {
    headers.push($(this).text().toLowerCase());
  });
  
  // Turn all existing rows into a loopable array
  $rows.each(function () {
    var $td = $(this).find('td');
    var h = {};
    
    // Use the headers from earlier to name our hash keys
    headers.forEach(function (header, i) {
      h[header] = $td.eq(i).text();   
    });
    
    data.push(h);
  });
  

  // Output the result
  $EXPORT.text(JSON.stringify(data));
});

$BTNGET.click(function () {
$.getJSON("reviews/next",
   function(data) {
     console.log(data);         
     addreview4(data)
   });

 });

$TRAINBTN.click(function () {
$.getJSON("/classifier/train",
function(data) {
     alert(data);         
     
   });
 });

$RESTARTBTN.click(function () {
$.getJSON("/review/reset",
function(data) {
     alert(data);         
     
   });
 });
function postData(address,data){
     $.ajax({
             type: "POST",
             url: address,
             data: JSON.stringify(data),
             contentType: "application/json; charset=utf-8",
             crossDomain: true,
             dataType: "json",
             success: function (data, status, jqXHR) {

                 console.log("successful posted")
             },

             error: function (jqXHR, status) {
                 // error handler
                 console.log(jqXHR);
                 alert('fail' + status.code);
             }
          });
}
