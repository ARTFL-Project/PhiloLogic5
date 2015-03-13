"use strict";

philoApp.controller('timeSeriesCtrl', ['$rootScope', '$location', 'URL', function($rootScope, $location, URL) {
    
    var vm = this;
    $rootScope.formData = angular.copy($location.search());
    if ($rootScope.formData.q === "" && $rootScope.report !== "bibliography") {
        $location.url(URL.objectToUrlString($rootScope.formData, {report: "bibliography"}));
    }
    if (typeof($rootScope.formData.year_interval) === "undefined") {
        var urlString = URL.objectToUrlString($rootScope.formData, {year_interval: $rootScope.philoConfig.time_series_intervals[0]});
        $location.url(urlString);
    }
    
    vm.percent = 0;
    vm.interval = parseInt($rootScope.formData.year_interval);
    
    vm.frequencyType = 'absolute_time';
    vm.toggleFrequency = function(frequencyType) {
        $('#time-series-buttons button').removeClass('active');
        $('#' + frequencyType).addClass('active');
        vm.frequencyType = frequencyType;
    }
    
    vm.hoverChart = function($event, title) {
        var element = $($event.currentTarget);
        element.popover('toggle')
    }
    
}]);